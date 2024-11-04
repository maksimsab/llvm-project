//===------------ SYCLUtils.h - SYCL utility functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
#define LLVM_TRANSFORMS_UTILS_SYCLUTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include <functional>
#include <string>
#include <vector>

namespace llvm {

constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";
constexpr char ATTR_SYCL_OPTLEVEL[] = "sycl-optlevel";

using CallGraphNodeAction = ::std::function<void(Function *)>;
using CallGraphFunctionFilter =
    std::function<bool(const Instruction *, const Function *)>;

// Traverses call graph starting from given function up the call chain applying
// given action to each function met on the way. If \c ErrorOnNonCallUse
// parameter is true, then no functions' uses are allowed except calls.
// Otherwise, any function where use of the current one happened is added to the
// call graph as if the use was a call.
// The 'functionFilter' parameter is a callback function that can be used to
// control which functions will be added to a call graph.
//
// The callback is invoked whenever a function being traversed is used
// by some instruction which is not a call to this instruction (e.g. storing
// function pointer to memory) - the first parameter is the using instructions,
// the second - the function being traversed. The parent function of the
// instruction is added to the call graph depending on whether the callback
// returns 'true' (added) or 'false' (not added).
// Functions which are part of the visited set ('Visited' parameter) are not
// traversed.

void traverseCallgraphUp(
    llvm::Function *F, CallGraphNodeAction NodeF,
    SmallPtrSetImpl<Function *> &Visited, bool ErrorOnNonCallUse,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; });

template <class CallGraphNodeActionF>
void traverseCallgraphUp(
    Function *F, CallGraphNodeActionF ActionF,
    SmallPtrSetImpl<Function *> &Visited, bool ErrorOnNonCallUse,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; }) {
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse, functionFilter);
}

template <class CallGraphNodeActionF>
void traverseCallgraphUp(
    Function *F, CallGraphNodeActionF ActionF, bool ErrorOnNonCallUse = true,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; }) {
  SmallPtrSet<Function *, 32> Visited;
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse, functionFilter);
}

inline bool isSYCLExternalFunction(const Function *F) {
  return F->hasFnAttribute(ATTR_SYCL_MODULE_ID);
}

/// Removes the global variable "llvm.used" and returns true on success.
/// "llvm.used" is a global constant array containing references to kernels
/// available in the module and callable from host code. The elements of
/// the array are ConstantExpr bitcast to i8*.
/// The variable must be removed as it is a) has done the job to the moment
/// of this function call and b) the references to the kernels callable from
/// host must not have users.
bool removeSYCLKernelsConstRefArray(Module &M);

using SYCLStringTable = std::vector<std::vector<std::string>>;

void writeSYCLStringTable(const SYCLStringTable &Table, raw_ostream &OS);

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYCLUTILS_H
