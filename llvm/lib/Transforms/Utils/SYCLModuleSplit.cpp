//===-------- SYCLModuleSplitter.cpp - split a module into callgraphs -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SYCLModuleSplit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerImpl.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SYCLUtils.h"

#include <algorithm>
#include <map>
#include <utility>
#include <variant>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "sycl_module_split"

namespace {
// Identifying name for global scope
constexpr char GLOBAL_SCOPE_NAME[] = "<GLOBAL>";
constexpr char SYCL_SCOPE_NAME[] = "<SYCL>";

EntryPointsGroupScope selectDeviceCodeGroupScope(const Module &M,
                                                 IRSplitMode Mode,
                                                 bool AutoSplitIsGlobalScope) {
  switch (Mode) {
  case IRSplitMode::IRSM_PER_TU:
    return Scope_PerModule;

  case IRSplitMode::IRSM_PER_KERNEL:
    return Scope_PerKernel;

  case IRSplitMode::IRSM_AUTO: {
    if (AutoSplitIsGlobalScope)
      return Scope_Global;

    // At the moment, we assume that per-source split is the best way of
    // splitting device code and can always be used except for cases handled
    // above.
    return Scope_PerModule;
  }

  case IRSplitMode::IRSM_NONE:
    return Scope_Global;
  }

  llvm_unreachable("unsupported split mode");
}

// Return true if the function is a SPIRV or SYCL builtin, e.g.
// _Z28__spirv_GlobalInvocationId_xv
bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.starts_with("__spirv_") || FName.starts_with("__sycl_");
}

// Return true if the function name starts with "__builtin_"
bool isGenericBuiltin(StringRef FName) {
  return FName.starts_with("__builtin_");
}

bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL ||
         F.getCallingConv() == CallingConv::AMDGPU_KERNEL;
}

bool isEntryPoint(const Function &F, bool EmitOnlyKernelsAsEntryPoints) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  if (isKernel(F))
    return true;

  if (!EmitOnlyKernelsAsEntryPoints) {
    // If not disabled, SYCL_EXTERNAL functions with sycl-module-id attribute
    // are also considered as entry points (except __spirv_* and __sycl_*
    // functions)
    return llvm::isSYCLExternalFunction(&F) &&
           !isSpirvSyclBuiltin(F.getName()) && !isGenericBuiltin(F.getName());
  }

  // Even if we are emitting only kernels as entry points, virtual functions
  // should still be treated as entry points, because they are going to be
  // outlined into separate device images and linked in later.
  return F.hasFnAttribute("indirectly-callable");
}

// Represents "dependency" or "use" graph of global objects (functions and
// global variables) in a module. It is used during device code split to
// understand which global variables and functions (other than entry points)
// should be included into a split module.
//
// Nodes of the graph represent LLVM's GlobalObjects, edges "A" -> "B" represent
// the fact that if "A" is included into a module, then "B" should be included
// as well.
//
// Examples of dependencies which are represented in this graph:
// - Function FA calls function FB
// - Function FA uses global variable GA
// - Global variable GA references (initialized with) function FB
// - Function FA stores address of a function FB somewhere
//
// The following cases are treated as dependencies between global objects:
// 1. Global object A is used within by a global object B in any way (store,
//    bitcast, phi node, call, etc.): "A" -> "B" edge will be added to the
//    graph;
// 2. function A performs an indirect call of a function with signature S and
//    there is a function B with signature S marked with "referenced-indirectly"
//    attribute. "A" -> "B" edge will be added to the graph;
class DependencyGraph {
public:
  using GlobalSet = SmallPtrSet<const GlobalValue *, 16>;

  DependencyGraph(const Module &M) {
    // Group functions by their signature to handle case (2) described above
    DenseMap<const FunctionType *, DependencyGraph::GlobalSet>
        FuncTypeToFuncsMap;
    for (const auto &F : M.functions()) {
      // Kernels can't be called (either directly or indirectly) in SYCL
      if (isKernel(F))
        continue;

      // Only functions which are marked with "referenced-indireclty" attribute
      // are considered to be indirect callee candidates.
      if (!F.hasFnAttribute("referenced-indirectly"))
        continue;

      FuncTypeToFuncsMap[F.getFunctionType()].insert(&F);
    }

    for (const auto &F : M.functions()) {
      // case (1), see comment above the class definition
      for (const Value *U : F.users())
        addUserToGraphRecursively(cast<const User>(U), &F);

      // case (2), see comment above the class definition
      for (const auto &I : instructions(F)) {
        const auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->isIndirectCall()) // Direct calls were handled above
          continue;

        // TODO: consider limiting set of potential callees to functions marked
        // with special attribute (like [[intel::device_indirectly_callable]])
        const FunctionType *Signature = CI->getFunctionType();
        // Note: strictly speaking, virtual functions are allowed to use
        // co-variant return types, i.e. we can actually miss a potential callee
        // here, because it has different signature (different return type).
        // However, this is not a problem for two reasons:
        // - opaque pointers will be enabled at some point and will make
        //   signatures the same in that case
        // - all virtual functions are referenced from vtable and therefore will
        //   anyway be preserved in a module
        const auto &PotentialCallees = FuncTypeToFuncsMap[Signature];
        Graph[&F].insert(PotentialCallees.begin(), PotentialCallees.end());
      }
    }

    // And every global variable (but their handling is a bit simpler)
    for (const auto &GV : M.globals())
      for (const Value *U : GV.users())
        addUserToGraphRecursively(cast<const User>(U), &GV);
  }

  iterator_range<GlobalSet::const_iterator>
  dependencies(const GlobalValue *Val) const {
    auto It = Graph.find(Val);
    return (It == Graph.end())
               ? make_range(EmptySet.begin(), EmptySet.end())
               : make_range(It->second.begin(), It->second.end());
  }

private:
  void addUserToGraphRecursively(const User *Root, const GlobalValue *V) {
    SmallVector<const User *, 8> WorkList;
    WorkList.push_back(Root);

    while (!WorkList.empty()) {
      const User *U = WorkList.pop_back_val();
      if (const auto *I = dyn_cast<const Instruction>(U)) {
        const auto *UFunc = I->getFunction();
        Graph[UFunc].insert(V);
      } else if (isa<const Constant>(U)) {
        if (const auto *GV = dyn_cast<const GlobalVariable>(U))
          Graph[GV].insert(V);
        // This could be a global variable or some constant expression (like
        // bitcast or gep). We trace users of this constant further to reach
        // global objects they are used by and add them to the graph.
        for (const auto *UU : U->users())
          WorkList.push_back(UU);
      } else
        llvm_unreachable("Unhandled type of function user");
    }
  }

  DenseMap<const GlobalValue *, GlobalSet> Graph;
  SmallPtrSet<const GlobalValue *, 1> EmptySet;
};

void collectFunctionsAndGlobalVariablesToExtract(
    SetVector<const GlobalValue *> &GVs, const Module &M,
    const EntryPointGroup &ModuleEntryPoints, const DependencyGraph &Deps,
    const std::function<bool(const Function *)> &IncludeFunctionPredicate =
        nullptr) {
  // We start with module entry points
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);

  // Non-discardable global variables are also include into the initial set
  for (const auto &GV : M.globals()) {
    if (!GV.isDiscardableIfUnused())
      GVs.insert(&GV);
  }

  // GVs has SetVector type. This type inserts a value only if it is not yet
  // present there. So, recursion is not expected here.
  size_t Idx = 0;
  while (Idx < GVs.size()) {
    const auto *Obj = GVs[Idx++];

    for (const GlobalValue *Dep : Deps.dependencies(Obj)) {
      if (const auto *Func = dyn_cast<const Function>(Dep)) {
        if (Func->isDeclaration())
          continue;

        // Functions can be additionally filtered
        if (!IncludeFunctionPredicate || IncludeFunctionPredicate(Func))
          GVs.insert(Func);
      } else {
        // Global variables are added unconditionally
        GVs.insert(Dep);
      }
    }
  }
}

ModuleDesc extractSubModule(const ModuleDesc &MD,
                            const SetVector<const GlobalValue *> GVs,
                            EntryPointGroup ModuleEntryPoints) {
  const Module &M = MD.getModule();
  // For each group of entry points collect all dependencies.
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  std::unique_ptr<Module> SubM = CloneModule(
      M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });
  // Replace entry points with cloned ones.
  EntryPointSet NewEPs;
  const EntryPointSet &EPs = ModuleEntryPoints.Functions;
  std::for_each(EPs.begin(), EPs.end(), [&](const Function *F) {
    NewEPs.insert(cast<Function>(VMap[F]));
  });
  ModuleEntryPoints.Functions = std::move(NewEPs);
  return ModuleDesc{std::move(SubM), std::move(ModuleEntryPoints)};
}

// The function produces a copy of input LLVM IR module M with only those
// functions and globals that can be called from entry points that are specified
// in ModuleEntryPoints vector, in addition to the entry point functions.
ModuleDesc extractCallGraph(const ModuleDesc &MD,
                            EntryPointGroup ModuleEntryPoints,
                            const DependencyGraph &CG,
                            const std::function<bool(const Function *)>
                                &IncludeFunctionPredicate = nullptr) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsAndGlobalVariablesToExtract(
      GVs, MD.getModule(), ModuleEntryPoints, CG, IncludeFunctionPredicate);

  ModuleDesc SplitM = extractSubModule(MD, GVs, std::move(ModuleEntryPoints));
  LLVM_DEBUG(SplitM.dump());
  SplitM.cleanup();

  return SplitM;
}

class ModuleCopier : public ModuleSplitterBase {
public:
  using ModuleSplitterBase::ModuleSplitterBase; // to inherit base constructors

  ModuleDesc nextSplit() override {
    ModuleDesc Desc{releaseInputModule(), nextGroup()};
    // Do some basic optimization like unused symbol removal
    // even if there was no split.
    Desc.cleanup();
    return Desc;
  }
};

class ModuleSplitter : public ModuleSplitterBase {
public:
  ModuleSplitter(ModuleDesc MD, EntryPointGroupVec GroupVec)
      : ModuleSplitterBase(std::move(MD), std::move(GroupVec)),
        CG(Input.getModule()) {}

  ModuleDesc nextSplit() override {
    return extractCallGraph(Input, nextGroup(), CG);
  }

private:
  DependencyGraph CG;
};

} // namespace

namespace llvm {

std::optional<IRSplitMode> convertStringToSplitMode(StringRef S) {
  static const StringMap<IRSplitMode> Values = {{"kernel", IRSplitMode::IRSM_PER_KERNEL},
                                                {"source", IRSplitMode::IRSM_PER_TU},
                                                {"auto", IRSplitMode::IRSM_AUTO},
                                                {"none", IRSplitMode::IRSM_NONE}};

  auto It = Values.find(S);
  if (It == Values.end())
    return std::nullopt;

  return It->second;
}

static void dumpEntryPoints(const EntryPointSet &C,
                     std::string_view Msg) {
  constexpr size_t INDENT = 4;
  dbgs().indent(INDENT) << "ENTRY POINTS"
                    << " " << Msg << " {\n";
  for (const Function *F : C)
    dbgs().indent(INDENT) << "  " << F->getName() << "\n";

  dbgs().indent(INDENT) << "}\n";
}

// Check "spirv.ExecutionMode" named metadata in the module and remove nodes
// that reference kernels that have dead prototypes or don't reference any
// kernel at all (nullptr). Dead prototypes are removed as well.
static void processSubModuleNamedMetadata(Module *M) {
  auto ExecutionModeMD = M->getNamedMetadata("spirv.ExecutionMode");
  if (!ExecutionModeMD)
    return;

  bool ContainsNodesToRemove = false;
  SmallVector<MDNode *, 128> ValueVec;
  for (auto Op : ExecutionModeMD->operands()) {
    assert(Op->getNumOperands() > 0);
    if (!Op->getOperand(0)) {
      ContainsNodesToRemove = true;
      continue;
    }

    // If the first operand is not nullptr then it has to be a kernel
    // function.
    Value *Val = cast<ValueAsMetadata>(Op->getOperand(0))->getValue();
    Function *F = cast<Function>(Val);
    // If kernel function is just a prototype and unused then we can remove it
    // and later remove corresponding spirv.ExecutionMode metadata node.
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      ContainsNodesToRemove = true;
      continue;
    }

    // Rememver nodes which we need to keep in the module.
    ValueVec.push_back(Op);
  }
  if (!ContainsNodesToRemove)
    return;

  if (ValueVec.empty()) {
    // If all nodes need to be removed then just remove named metadata
    // completely.
    ExecutionModeMD->eraseFromParent();
  } else {
    ExecutionModeMD->clearOperands();
    for (auto MD : ValueVec)
      ExecutionModeMD->addOperand(MD);
  }
}

void ModuleDesc::cleanup() {
  // Externalize them so they are not dropped by GlobalDCE
  for (Function &F : *M)
    if (F.hasFnAttribute("indirectly-callable"))
      F.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  // Do cleanup.
  MPM.addPass(GlobalDCEPass());           // Delete unreachable globals.
  MPM.addPass(StripDeadDebugInfoPass());  // Remove dead debug info.
  MPM.addPass(StripDeadPrototypesPass()); // Remove dead func decls.
  MPM.run(*M, MAM);

  // Original module may have named metadata (spirv.ExecutionMode) referencing
  // kernels in the module. Some of the Metadata nodes may reference kernels
  // which are not included into the extracted submodule, in such case
  // CloneModule either leaves that metadata nodes as is but they will reference
  // dead prototype of the kernel or operand will be replace with nullptr. So
  // process all nodes in the named metadata and remove nodes which are
  // referencing kernels which are not included into submodule.
  processSubModuleNamedMetadata(M.get());
}

ModuleDesc ModuleDesc::clone() const {
  std::unique_ptr<Module> NewModule = CloneModule(getModule());
  ModuleDesc NewMD(std::move(NewModule));
  NewMD.EntryPoints.Props = EntryPoints.Props;
  return NewMD;
}

void ModuleDesc::dump() const {
  assert(M && "dump of empty ModuleDesc");
  dbgs() << "split_module::ModuleDesc[" << M->getName() << "] {\n";
  dumpEntryPoints(entries(), EntryPoints.GroupId.c_str());
  dbgs() << "}\n";
}

void EntryPointGroup::saveNames(std::vector<std::string> &Dest) const {
  Dest.reserve(Dest.size() + Functions.size());
  std::transform(Functions.begin(), Functions.end(),
                 std::inserter(Dest, Dest.end()),
                 [](const Function *F) { return F->getName().str(); });
}

void EntryPointGroup::rebuildFromNames(const std::vector<std::string> &Names,
                                       const Module &M) {
  Functions.clear();
  auto It0 = Names.cbegin();
  auto It1 = Names.cend();
  std::for_each(It0, It1, [&](const std::string &Name) {
    // Sometimes functions considered entry points (those for which isEntryPoint
    // returned true) may be dropped by optimizations, such as AlwaysInliner.
    // For example, if a linkonce_odr function is inlined and there are no other
    // uses, AlwaysInliner drops it. It is responsibility of the user to make an
    // entry point not have internal linkage (such as linkonce_odr) to guarantee
    // its availability in the resulting device binary image.
    if (Function *F = M.getFunction(Name))
      Functions.insert(F);
  });
}

void EntryPointGroup::rebuild(const Module &M) {
  for (const Function &F : M.functions())
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      Functions.insert(const_cast<Function *>(&F));
}

std::string ModuleDesc::makeSymbolTable() const {
  std::string ST;
  for (const Function *F : EntryPoints.Functions)
    ST += (Twine(F->getName()) + "\n").str();

  return ST;
}

namespace {
// This is a helper class, which allows to group/categorize function based on
// provided rules. It is intended to be used in device code split
// implementation.
//
// "Rule" is a simple routine, which returns a string for an llvm::Function
// passed to it. There could be more than one rule and they are applied in order
// of their registration. Results obtained from those rules are concatenated
// together to produce the final result.
//
// There are some predefined rules for the most popular use-cases, like grouping
// functions together based on an attribute value or presence of a metadata.
// However, there is also a possibility to register a custom callback function
// as a rule, to implement custom/more complex logic.
class FunctionsCategorizer {
public:
  FunctionsCategorizer() = default;

  std::string computeCategoryFor(Function *) const;

  // Accepts a callback, which should return a string based on provided
  // function, which will be used as an entry points group identifier.
  void registerRule(const std::function<std::string(Function *)> &Callback) {
    Rules.emplace_back(Rule::RKind::K_Callback, Callback);
  }

  // Creates a simple rule, which adds a value of a string attribute into a
  // resulting identifier.
  void registerSimpleStringAttributeRule(StringRef AttrName) {
    Rules.emplace_back(Rule::RKind::K_SimpleStringAttribute, AttrName);
  }

  // Creates a simple rule, which adds a value of a string metadata into a
  // resulting identifier.
  void registerSimpleStringMetadataRule(StringRef MetadataName) {
    Rules.emplace_back(Rule::RKind::K_SimpleStringMetadata, MetadataName);
  }

  // Creates a simple rule, which adds one or another value to a resulting
  // identifier based on the presence of a metadata on a function.
  void registerSimpleFlagAttributeRule(StringRef AttrName,
                                       StringRef IfPresentStr,
                                       StringRef IfAbsentStr = "") {
    Rules.emplace_back(Rule::RKind::K_FlagAttribute,
                       Rule::FlagRuleData{AttrName, IfPresentStr, IfAbsentStr});
  }

  // Creates a simple rule, which adds one or another value to a resulting
  // identifier based on the presence of a metadata on a function.
  void registerSimpleFlagMetadataRule(StringRef MetadataName,
                                      StringRef IfPresentStr,
                                      StringRef IfAbsentStr = "") {
    Rules.emplace_back(
        Rule::RKind::K_FlagMetadata,
        Rule::FlagRuleData{MetadataName, IfPresentStr, IfAbsentStr});
  }

  // Creates a rule, which adds a list of dash-separated integers converted
  // into strings listed in a metadata to a resulting identifier.
  void registerListOfIntegersInMetadataRule(StringRef MetadataName) {
    Rules.emplace_back(Rule::RKind::K_IntegersListMetadata, MetadataName);
  }

  // Creates a rule, which adds a list of sorted dash-separated integers
  // converted into strings listed in a metadata to a resulting identifier.
  void registerListOfIntegersInMetadataSortedRule(StringRef MetadataName) {
    Rules.emplace_back(Rule::RKind::K_SortedIntegersListMetadata, MetadataName);
  }

private:
  struct Rule {
    struct FlagRuleData {
      StringRef Name, IfPresentStr, IfAbsentStr;
    };

  private:
    std::variant<StringRef, FlagRuleData,
                 std::function<std::string(Function *)>>
        Storage;

  public:
    enum class RKind {
      // Custom callback function
      K_Callback,
      // Copy value of the specified attribute, if present
      K_SimpleStringAttribute,
      // Copy value of the specified metadata, if present
      K_SimpleStringMetadata,
      // Use one or another string based on the specified metadata presence
      K_FlagMetadata,
      // Use one or another string based on the specified attribute presence
      K_FlagAttribute,
      // Concatenate and use list of integers from the specified metadata
      K_IntegersListMetadata,
      // Sort, concatenate and use list of integers from the specified metadata
      K_SortedIntegersListMetadata
    };
    RKind Kind;

    // Returns an index into std::variant<...> Storage defined above, which
    // corresponds to the specified rule Kind.
    constexpr static std::size_t storage_index(RKind K) {
      switch (K) {
      case RKind::K_SimpleStringAttribute:
      case RKind::K_IntegersListMetadata:
      case RKind::K_SimpleStringMetadata:
      case RKind::K_SortedIntegersListMetadata:
        return 0;
      case RKind::K_Callback:
        return 2;
      case RKind::K_FlagMetadata:
      case RKind::K_FlagAttribute:
        return 1;
      }
      // can't use llvm_unreachable in constexpr context
      return std::variant_npos;
    }

    template <RKind K> auto getStorage() const {
      return std::get<storage_index(K)>(Storage);
    }

    template <typename... Args>
    Rule(RKind K, Args... args) : Storage(args...), Kind(K) {
      assert(storage_index(K) == Storage.index());
    }

    Rule(Rule &&Other) = default;
  };

  SmallVector<Rule, 0> Rules;
};

std::string FunctionsCategorizer::computeCategoryFor(Function *F) const {
  SmallString<256> Result;
  for (const auto &R : Rules) {
    StringRef AttrName;
    StringRef MetadataName;
    Rule::FlagRuleData Data;

    switch (R.Kind) {
    case Rule::RKind::K_Callback:
      Result += R.getStorage<Rule::RKind::K_Callback>()(F);
      break;

    case Rule::RKind::K_SimpleStringAttribute:
      AttrName = R.getStorage<Rule::RKind::K_SimpleStringAttribute>();
      if (F->hasFnAttribute(AttrName)) {
        Attribute Attr = F->getFnAttribute(AttrName);
        Result += Attr.getValueAsString();
      }
      break;

    case Rule::RKind::K_SimpleStringMetadata:
      MetadataName = R.getStorage<Rule::RKind::K_SimpleStringMetadata>();
      if (F->hasMetadata(MetadataName)) {
        auto *MDN = F->getMetadata(MetadataName);
        for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
          MDString *S = cast<llvm::MDString>(MDN->getOperand(I).get());
          Result += "-" + S->getString().str();
        }
      }
      break;

    case Rule::RKind::K_FlagMetadata:
      Data = R.getStorage<Rule::RKind::K_FlagMetadata>();
      if (F->hasMetadata(Data.Name))
        Result += Data.IfPresentStr;
      else
        Result += Data.IfAbsentStr;
      break;

    case Rule::RKind::K_IntegersListMetadata:
      MetadataName = R.getStorage<Rule::RKind::K_IntegersListMetadata>();
      if (F->hasMetadata(MetadataName)) {
        auto *MDN = F->getMetadata(MetadataName);
        for (const MDOperand &MDOp : MDN->operands())
          Result +=
              "-" + std::to_string(
                        mdconst::extract<ConstantInt>(MDOp)->getZExtValue());
      }
      break;

    case Rule::RKind::K_SortedIntegersListMetadata:
      MetadataName = R.getStorage<Rule::RKind::K_IntegersListMetadata>();
      if (F->hasMetadata(MetadataName)) {
        MDNode *MDN = F->getMetadata(MetadataName);

        SmallVector<std::uint64_t, 8> Values;
        for (const MDOperand &MDOp : MDN->operands())
          Values.push_back(mdconst::extract<ConstantInt>(MDOp)->getZExtValue());

        llvm::sort(Values);

        for (std::uint64_t V : Values)
          Result += "-" + std::to_string(V);
      }
      break;

    case Rule::RKind::K_FlagAttribute:
      Data = R.getStorage<Rule::RKind::K_FlagAttribute>();
      if (F->hasFnAttribute(Data.Name))
        Result += Data.IfPresentStr;
      else
        Result += Data.IfAbsentStr;
      break;
    }

    Result += "-";
  }

  return static_cast<std::string>(Result);
}
} // namespace

std::unique_ptr<ModuleSplitterBase>
getDeviceCodeSplitter(ModuleDesc MD, IRSplitMode Mode, bool IROutputOnly,
                      bool EmitOnlyKernelsAsEntryPoints) {
  FunctionsCategorizer Categorizer;

  EntryPointsGroupScope Scope =
      selectDeviceCodeGroupScope(MD.getModule(), Mode, IROutputOnly);

  switch (Scope) {
  case Scope_Global:
    // We simply perform entry points filtering, but group all of them together.
    Categorizer.registerRule(
        [](Function *) -> std::string { return GLOBAL_SCOPE_NAME; });
    break;
  case Scope_PerKernel:
    // Per-kernel split is quite simple: every kernel goes into a separate
    // module and that's it, no other rules required.
    Categorizer.registerRule(
        [](Function *F) -> std::string { return F->getName().str(); });
    break;
  case Scope_PerModule:
    // The most complex case, because we should account for many other features
    // like aspects used in a kernel, large-grf mode, reqd-work-group-size, etc.

    // This is core of per-source device code split
    Categorizer.registerSimpleStringAttributeRule(ATTR_SYCL_MODULE_ID);

    // This attribute marks virtual functions and effectively dictates how they
    // should be groupped together. By design we won't split those groups of
    // virtual functions further even if functions from the same group use
    // different optional features and therefore this rule is put here.
    // Strictly speaking, we don't even care about module-id splitting for
    // those, but to avoid that we need to refactor the whole categorizer.
    // However, this is good enough as it is for an initial version.
    // TODO: for AOT use case we shouldn't be outlining those and instead should
    // only select those functions which are compatible with the target device
    Categorizer.registerSimpleStringAttributeRule("indirectly-callable");

    // Optional features
    // Note: Add more rules at the end of the list to avoid chaning orders of
    // output files in existing tests.
    Categorizer.registerSimpleStringAttributeRule("sycl-register-alloc-mode");
    Categorizer.registerSimpleStringAttributeRule("sycl-grf-size");
    Categorizer.registerListOfIntegersInMetadataRule("reqd_work_group_size");
    Categorizer.registerListOfIntegersInMetadataRule("work_group_num_dim");
    Categorizer.registerListOfIntegersInMetadataRule(
        "intel_reqd_sub_group_size");
    Categorizer.registerSimpleStringAttributeRule(ATTR_SYCL_OPTLEVEL);
    break;
  }

  // std::map is used here to ensure stable ordering of entry point groups,
  // which is based on their contents, this greatly helps LIT tests
  std::map<std::string, EntryPointSet> EntryPointsMap;

  // Only process module entry points:
  for (auto &F : MD.getModule().functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;

    std::string Key = Categorizer.computeCategoryFor(&F);
    EntryPointsMap[std::move(Key)].insert(&F);
  }

  EntryPointGroupVec Groups;
  if (EntryPointsMap.empty()) {
    // No entry points met, record this.
    Groups.emplace_back(GLOBAL_SCOPE_NAME, EntryPointSet{});
  } else {
    Groups.reserve(EntryPointsMap.size());
    // Start with properties of a source module
    EntryPointGroup::Properties MDProps = MD.getEntryPointGroup().Props;
    for (auto &[Key, EntryPoints] : EntryPointsMap)
      Groups.emplace_back(Key, std::move(EntryPoints), MDProps);
  }

  bool DoSplit = (Mode != IRSplitMode::IRSM_NONE &&
                  (Groups.size() > 1 || !Groups.begin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups));

  return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups));
}

static Error saveModuleIRInFile(Module &M, StringRef FilePath,
                                bool OutputAssembly) {
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(FilePath, FD))
    return errorCodeToError(EC);

  raw_fd_ostream OS(FD, true);
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(OS));
  else
    MPM.addPass(BitcodeWriterPass(OS));

  MPM.run(M, MAM);
  return Error::success();
}

static Expected<SYCLSplitModule>
saveModuleDesc(ModuleDesc &MD, std::string Prefix, bool OutputAssembly) {
  SYCLSplitModule SM;
  Prefix += OutputAssembly ? ".ll" : ".bc";
  Error E = saveModuleIRInFile(MD.getModule(), Prefix, OutputAssembly);
  if (E)
    return E;

  SM.ModuleFilePath = Prefix;
  SM.Symbols = MD.makeSymbolTable();
  return SM;
}

Expected<SmallVector<SYCLSplitModule, 0>>
parseSYCLSplitModulesFromFile(StringRef File) {
  auto EntriesMBOrErr = llvm::MemoryBuffer::getFile(File);
  if (!EntriesMBOrErr)
    return createFileError(File, EntriesMBOrErr.getError());

  line_iterator LI(**EntriesMBOrErr);
  if (LI.is_at_eof() || *LI != "[Code|Symbols]")
    return createStringError(inconvertibleErrorCode(),
                             "invalid SYCL Table file.");

  // "Code" and "Symbols" at the moment.
  static constexpr int NUMBER_COLUMNS = 2;
  ++LI;
  SmallVector<SYCLSplitModule, 0> Modules;
  while (!LI.is_at_eof()) {
    StringRef Line = *LI;
    if (Line.empty())
      return createStringError("invalid SYCL table row.");

    SmallVector<StringRef, NUMBER_COLUMNS> Parts;
    Line.split(Parts, "|");
    if (Parts.size() != NUMBER_COLUMNS)
      return createStringError("invalid SYCL Table row.");

    auto [IRFilePath, SymbolsFilePath] = std::tie(Parts[0], Parts[1]);
    if (SymbolsFilePath.empty())
      return createStringError("invalid SYCL Table row.");

    auto MBOrErr = MemoryBuffer::getFile(SymbolsFilePath);
    if (!MBOrErr)
      return createFileError(SymbolsFilePath, MBOrErr.getError());

    auto &MB2 = *MBOrErr;
    std::string Symbols =
        std::string(MB2->getBufferStart(), MB2->getBufferEnd());
    Modules.emplace_back(IRFilePath, std::move(Symbols));
    ++LI;
  }

  return Modules;
}

Expected<SmallVector<SYCLSplitModule, 0>>
splitSYCLModule(std::unique_ptr<Module> M, ModuleSplitterSettings Settings) {
  ModuleDesc MD = std::move(M);
  auto Splitter = getDeviceCodeSplitter(std::move(MD), Settings.Mode,
                                        /*IROutputOnly=*/false,
                                        /*EmitOnlyKernelsAsEntryPoints=*/false);

  size_t ID = 0;
  SmallVector<SYCLSplitModule, 0> OutputImages;
  while (Splitter->hasMoreSplits()) {
    ModuleDesc MD = Splitter->nextSplit();

    std::string OutIRFileName = (Settings.OutputPrefix + "_" + Twine(ID)).str();
    auto SplitImageOrErr =
        saveModuleDesc(MD, OutIRFileName, Settings.OutputAssembly);
    if (!SplitImageOrErr)
      return SplitImageOrErr.takeError();

    OutputImages.emplace_back(std::move(*SplitImageOrErr));
    ++ID;
  }

  return OutputImages;
}

} // namespace llvm
