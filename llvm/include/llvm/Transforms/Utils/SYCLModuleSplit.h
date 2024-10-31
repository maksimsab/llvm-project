//===-------- SYCLModuleSplit.h - module split ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module into call graphs. A callgraph here is a set
// of entry points with all functions reachable from them via a call. The result
// of the split is new modules containing corresponding callgraph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_MODULE_SPLIT_H
#define LLVM_SYCL_MODULE_SPLIT_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

// TODO(maksimsab):
//  * Maybe fix doxygen comments.

namespace llvm {

class Function;
class Module;

enum class IRSplitMode {
  IRSM_PER_TU,           // one module per translation unit
  IRSM_PER_KERNEL,       // one module per kernel
  IRSM_AUTO,             // automatically select split mode
  IRSM_NONE              // no splitting
};

/// \returns IRSplitMode value if \p S is recognized. Otherwise, std::nullopt is
/// returned.
std::optional<IRSplitMode> convertStringToSplitMode(StringRef S);

// A vector that contains all entry point functions in a split module.
using EntryPointSet = SetVector<Function *>;

/// Describes scope covered by each entry in the module-entry points map
/// populated by the groupEntryPointsByScope function.
enum EntryPointsGroupScope {
  Scope_PerKernel, // one entry per kernel
  Scope_PerModule, // one entry per module
  Scope_Global     // single entry in the map for all kernels
};

/// Represents a named group of device code entry points - kernels and
/// SYCL_EXTERNAL functions.
struct EntryPointGroup {
  // Properties an entry point (EP) group
  struct Properties {
    // Scope represented by EPs in a group
    EntryPointsGroupScope Scope = Scope_Global;
  };

  std::string GroupId;
  EntryPointSet Functions;
  Properties Props;

  EntryPointGroup(StringRef GroupId = "") : GroupId(GroupId) {}
  EntryPointGroup(StringRef GroupId, EntryPointSet Functions)
      : GroupId(GroupId), Functions(std::move(Functions)) {}
  EntryPointGroup(StringRef GroupId, EntryPointSet Functions,
                  const Properties &Props)
      : GroupId(GroupId), Functions(std::move(Functions)), Props(Props) {}

  void saveNames(std::vector<std::string> &Dest) const;
  void rebuildFromNames(const std::vector<std::string> &Names, const Module &M);
  void rebuild(const Module &M);
};

using EntryPointGroupVec = SmallVector<EntryPointGroup, 0>;

/// Annotates an llvm::Module with information necessary to perform and track
/// result of device code (llvm::Module instances) splitting:
/// - entry points of the module determined e.g. by a module splitter, as well
///   as information about entry point origin (e.g. result of a scoped split)
/// - its properties, such as whether it has specialization constants uses
/// It also provides convenience functions for entry point set transformation
/// between llvm::Function object and string representations.
class ModuleDesc {
  std::unique_ptr<Module> M;
  EntryPointGroup EntryPoints;

public:
  ModuleDesc(std::unique_ptr<Module> M) : M(std::move(M)) {}

  ModuleDesc(std::unique_ptr<Module> M, EntryPointGroup EntryPoints)
      : M(std::move(M)), EntryPoints(std::move(EntryPoints)) {}

  ModuleDesc(std::unique_ptr<Module> M, const std::vector<std::string> &Names)
      : M(std::move(M)) {
    rebuildEntryPoints(Names);
  }

  const EntryPointSet &entries() const { return EntryPoints.Functions; }
  const EntryPointGroup &getEntryPointGroup() const { return EntryPoints; }
  EntryPointSet &entries() { return EntryPoints.Functions; }
  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }
  std::unique_ptr<Module> releaseModulePtr() { return std::move(M); }

  // Sometimes, during module transformations, some Function objects within the
  // module are replaced with different Function objects with the same name.
  // Entry points need to be updated to include the replacement function.
  // save/rebuild pair of functions is provided to automate this process.
  void saveEntryPointNames(std::vector<std::string> &Dest) {
    EntryPoints.saveNames(Dest);
  }

  void rebuildEntryPoints(const std::vector<std::string> &Names) {
    EntryPoints.rebuildFromNames(Names, getModule());
  }

  void rebuildEntryPoints(const Module &M) { EntryPoints.rebuild(M); }

  void rebuildEntryPoints() { EntryPoints.rebuild(*M); }

  // Cleans up module IR - removes dead globals, debug info etc.
  void cleanup();

  ModuleDesc clone() const;

  std::string makeSymbolTable() const;

  void dump() const;
};

/// Module split support interface.
/// It gets a module (in a form of module descriptor, to get additional info) and
/// a collection of entry points groups. Each group specifies subset entry points
// from input module that should be included in a split module.
class ModuleSplitterBase {
protected:
  ModuleDesc Input;
  EntryPointGroupVec Groups;

protected:
  EntryPointGroup nextGroup() {
    assert(hasMoreSplits() && "Reached end of entry point groups list.");
    EntryPointGroup Res = std::move(Groups.back());
    Groups.pop_back();
    return Res;
  }

  Module &getInputModule() { return Input.getModule(); }

  std::unique_ptr<Module> releaseInputModule() {
    return Input.releaseModulePtr();
  }

public:
  ModuleSplitterBase(ModuleDesc MD, EntryPointGroupVec GroupVec)
      : Input(std::move(MD)), Groups(std::move(GroupVec)) {
    assert(!Groups.empty() && "Entry points groups collection is empty!");
  }

  virtual ~ModuleSplitterBase() = default;

  /// Gets next subsequence of entry points in an input module and provides split
  /// submodule containing these entry points and their dependencies.
  virtual ModuleDesc nextSplit() = 0;

  /// Returns a number of remaining modules, which can be split out using this
  /// splitter. The value is reduced by 1 each time nextSplit is called.
  size_t remainingSplits() const { return Groups.size(); }

  /// Check that there are still submodules to split.
  bool hasMoreSplits() const { return remainingSplits() > 0; }
};

std::unique_ptr<ModuleSplitterBase>
getDeviceCodeSplitter(ModuleDesc MD, IRSplitMode Mode, bool IROutputOnly,
                      bool EmitOnlyKernelsAsEntryPoints);

/// The structure represents a split LLVM Module accompanied by additional information.
/// Split Modules are being stored at disk due to the high RAM consumption during the whole splitting process.
struct SYCLSplitModule {
  std::string ModuleFilePath;
  std::string Symbols;

  SYCLSplitModule() = default;
  SYCLSplitModule(const SYCLSplitModule &) = default;
  SYCLSplitModule &operator=(const SYCLSplitModule &) = default;
  SYCLSplitModule(SYCLSplitModule &&) = default;
  SYCLSplitModule &operator=(SYCLSplitModule &&) = default;

  SYCLSplitModule(std::string_view File, std::string Symbols)
      : ModuleFilePath(File), Symbols(std::move(Symbols)) {}
};

struct ModuleSplitterSettings {
  IRSplitMode Mode;
  bool OutputAssembly = false; // Bitcode or LLVM IR.
  StringRef OutputPrefix;
};

/// Parses the string table.
Expected<SmallVector<SYCLSplitModule, 0>>
parseSYCLSplitModulesFromFile(StringRef File);

/// Splits the given module \p M according to the given \p Settings.
Expected<SmallVector<SYCLSplitModule, 0>>
splitSYCLModule(std::unique_ptr<Module> M, ModuleSplitterSettings Settings);

} // namespace llvm

#endif // LLVM_SYCL_MODULE_SPLIT_H
