//===-------- SplitModuleByCategory.h - module split ------------*- C++ -*-===//
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module by categories.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H
#define LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H

#include "llvm/ADT/STLFunctionalExtras.h"

#include <memory>
#include <optional>
#include <string>

namespace llvm {

class Module;
class Function;

/// FunctionCategorizer returns integer category for the given Function.
/// Otherwise, it returns std::nullopt if function doesn't have a category.
using FunctionCategorizer = function_ref<std::optional<int>(const Function &F)>;

using PostSplitCallbackType = function_ref<void(std::unique_ptr<Module> Part)>;

/// Splits the given module \p M by categories calculated by the given \p FC.
/// Some functions have their category by which they are being split.
/// The result of the split is new modules containing call graphs with
/// categorized functions and all functions reachable by call.
/// Every split image is being passed to \p Callback for further possible
/// processing.
void splitModuleByCategory(std::unique_ptr<Module> M, FunctionCategorizer FC,
                           PostSplitCallbackType Callback);

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SYCLSPLITMODULE_H
