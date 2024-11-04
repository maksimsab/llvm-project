//===------------ SYCLUtils.cpp - SYCL utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SYCL utility functions.
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Utils/SYCLUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"

namespace llvm {

void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction ActionF,
                         SmallPtrSetImpl<Function *> &FunctionsVisited,
                         bool ErrorOnNonCallUse,
                         const CallGraphFunctionFilter &functionFilter) {
  SmallVector<Function *, 32> Worklist;

  if (FunctionsVisited.count(F) == 0)
    Worklist.push_back(F);

  while (!Worklist.empty()) {
    Function *CurF = Worklist.pop_back_val();
    FunctionsVisited.insert(CurF);
    // Apply the action function.
    ActionF(CurF);

    // Update all callers as well.
    for (auto It = CurF->use_begin(); It != CurF->use_end(); It++) {
      auto FCall = It->getUser();
      auto ErrMsg =
          llvm::Twine(__FILE__ " ") +
          "Function use other than call detected while traversing call\n"
          "graph up to a kernel";
      if (!isa<CallInst>(FCall)) {
        // A use other than a call is met...
        if (ErrorOnNonCallUse) {
          // ... non-call is an error - report
          llvm::report_fatal_error(ErrMsg);
        } else {
          // ... non-call is OK - add using function to the worklist
          if (auto *I = dyn_cast<Instruction>(FCall)) {
            if (!functionFilter(I, CurF)) {
              continue;
            }

            auto UseF = I->getFunction();

            if (FunctionsVisited.count(UseF) == 0) {
              Worklist.push_back(UseF);
            }
          }
        }
      } else {
        auto *CI = cast<CallInst>(FCall);

        if ((CI->getCalledFunction() != CurF)) {
          // CurF is used in a call, but not as the callee.
          if (ErrorOnNonCallUse)
            llvm::report_fatal_error(ErrMsg);
        } else {
          auto FCaller = CI->getFunction();

          if (!FunctionsVisited.count(FCaller)) {
            Worklist.push_back(FCaller);
          }
        }
      }
    }
  }
}

bool removeSYCLKernelsConstRefArray(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");

  if (!GV)
    return false;

  assert(GV->user_empty() && "Unexpected llvm.used users");
  Constant *Initializer = GV->getInitializer();
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer and all operands of it.
  SmallVector<Constant *, 8> IOperands;
  for (auto It = Initializer->op_begin(); It != Initializer->op_end(); It++)
    IOperands.push_back(cast<Constant>(*It));
  assert(llvm::isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of llvm.used global");
  Initializer->destroyConstant();
  for (auto It = IOperands.begin(); It != IOperands.end(); It++) {
    auto Op = (*It)->stripPointerCasts();
    auto *F = dyn_cast<Function>(Op);
    if (llvm::isSafeToDestroyConstant(*It))
      (*It)->destroyConstant();
    else if (F && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
             !F->use_empty()) {
      // The element in "llvm.used" array has other users. That is Ok for
      // specialization constants, but is wrong for kernels.
      llvm::report_fatal_error("Unexpected usage of SYCL kernel");
    }

    // Remove unused kernel declarations to avoid LLVM IR check fails.
    if (F && F->isDeclaration() && F->use_empty())
      F->eraseFromParent();
  }

  return true;
}

void writeSYCLStringTable(const SYCLStringTable &Table, raw_ostream &OS) {
  assert(Table.size() > 0 && "table should contain at least column titles");
  size_t numberColumns = Table[0].size();
  assert(numberColumns > 0 && "table should be non-empty");
  OS << '[' << join(Table[0].begin(), Table[0].end(), "|") << "]\n";
  for (size_t I = 1, E = Table.size(); I != E; ++I) {
    assert(Table[I].size() == numberColumns && "row's size should be equal");
    OS << join(Table[I].begin(), Table[I].end(), "|") << '\n';
  }
}

} // namespace llvm
