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

namespace llvm {

class Function;
class Module;

enum class IRSplitMode {
  IRSM_PER_TU,     // one module per translation unit
  IRSM_PER_KERNEL, // one module per kernel
  IRSM_AUTO,       // automatically select split mode
  IRSM_NONE        // no splitting
};

/// \returns IRSplitMode value if \p S is recognized. Otherwise, std::nullopt is
/// returned.
std::optional<IRSplitMode> convertStringToSplitMode(StringRef S);

// A vector that contains all entry point functions in a split module.
using EntryPointSet = SetVector<const Function *>;

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
};

// TODO: move it into cpp file.
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

  const EntryPointSet &entries() const { return EntryPoints.Functions; }
  const EntryPointGroup &getEntryPointGroup() const { return EntryPoints; }
  EntryPointSet &entries() { return EntryPoints.Functions; }
  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }
  std::unique_ptr<Module> releaseModulePtr() { return std::move(M); }

  // Cleans up module IR - removes dead globals, debug info etc.
  void cleanup();

  std::string makeSymbolTable() const;

  void dump() const;
};

/// The structure represents a split LLVM Module accompanied by additional
/// information. Split Modules are being stored at disk due to the high RAM
/// consumption during the whole splitting process.
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
