//===- sycl-post-link.cpp - SYCL post-link device code processing tool ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This source is a collection of utilities run on device code's LLVM IR before
// handing off to back-end for further compilation or emitting SPIRV. The
// utilities are:
// - module splitter to split a big input module into smaller ones
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/SYCLModuleSplit.h"
#include "llvm/Transforms/Utils/SYCLUtils.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::sycl::utils;

using string_vector = std::vector<std::string>;

namespace {

void error(const Twine &Msg) {
  errs() << "sycl-post-link: " << Msg << '\n';
  exit(1);
}

void checkError(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// InputFilename - The filename to read from.
cl::opt<std::string> InputFilename{cl::Positional,
                                   cl::desc("<input bitcode file>"),
                                   cl::init("-"), cl::value_desc("filename")};

cl::opt<std::string> OutputDir{
    "out-dir",
    cl::desc(
        "Directory where files listed in the result file table will be output"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

cl::opt<std::string> Output{"o", cl::desc("Output path"), cl::value_desc("output filename"), cl::cat(PostLinkCat)};

cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                    cl::cat(PostLinkCat)};

cl::opt<bool> IROutputOnly{"ir-output-only", cl::desc("Output single IR file"),
                           cl::cat(PostLinkCat)};

cl::opt<bool> OutputAssembly{"S", cl::desc("Write output as LLVM assembly"),
                             cl::Hidden, cl::cat(PostLinkCat)};

cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional,
    cl::init(SPLIT_NONE),
    cl::values(clEnumValN(SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel"),
               clEnumValN(SPLIT_AUTO, "auto",
                          "Choose split mode automatically")),
    cl::cat(PostLinkCat));

using StringTable = std::vector<std::vector<std::string>>;

void writeStringTable(const StringTable &Table, raw_ostream &OS) {
  assert(Table.size() > 0 && "table should contain at least column titles");
  OS << '[' << join(Table[0].begin(), Table[0].end(), "|") << "]\n";
  for (size_t I = 1, E = Table.size(); I != E; ++I)
    OS << join(Table[I].begin(), Table[I].end(), "|") << '\n';
}

Expected<StringTable>
processInputModule(std::unique_ptr<Module> M) {
  // After linking device bitcode "llvm.used" holds references to the kernels
  // that are defined in the device image. But after splitting device image into
  // separate kernels we may end up with having references to kernel declaration
  // originating from "llvm.used" in the IR that is passed to llvm-spirv tool,
  // and these declarations cause an assertion in llvm-spirv. To workaround this
  // issue remove "llvm.used" from the input module before performing any other
  // actions.
  removeSYCLKernelsConstRefArray(*M);

  // -ir-output-only assumes single module output thus no code splitting.
  // Violation of this invariant is user error and must've been reported.
  // However, if split mode is "auto", then entry point filtering is still
  // performed.
  assert((!IROutputOnly || (SplitMode == module_split::SPLIT_NONE) ||
          (SplitMode == module_split::SPLIT_AUTO)) &&
         "invalid split mode for IR-only output");

  ModuleSplitterSettings Settings;
  Settungs.SplitMode = SplitMode;
  Settings.OutputAssembly = OutputAssembly;
  Settings.OutputPrefix = Output;
  auto ModulesOrErr = splitSYCLModule(std::move(M), Settings);
  if (!ModulesOrErr)
    return ModulesOrErr.takeError();

  std::vector<SplitModule> &Modules = *ModulesOrErr;
  // Construct the resulting table which will accumulate all the outputs.
  std::vector<std::string> ColumnTitles = {"Code"};
  StringTable Table = {ColumnTitles};
  for (const SplitModule &SM : Modules)
    Table.push_back({SM.ModuleFilePath});

  return Table;
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};

  LLVMContext Context;
  cl::ParseCommandLineOptions(
      argc, argv,
      "SYCL post-link device code processing tool.\n"
      "This is a collection of utilities run on device code's LLVM IR before\n"
      "handing off to back-end for further compilation or emitting SPIRV.\n"
      "The utilities are:\n"
      "- SYCL and ESIMD kernels can be split into separate modules with\n"
      "  '-split-esimd' option. The option has no effect when there is only\n"
      "  one type of kernels in the input module. Functions unreachable from\n"
      "  any entry point (kernels and SYCL_EXTERNAL functions) are\n"
      "  dropped from the resulting module(s).\n"
      "- Module splitter to split a big input module into smaller ones.\n"
      "  Groups kernels using function attribute 'sycl-module-id', i.e.\n"
      "  kernels with the same values of the 'sycl-module-id' attribute will\n"
      "  be put into the same module. If -split=kernel option is specified,\n"
      "  one module per kernel will be emitted.\n"
      "  '-split=auto' mode automatically selects the best way of splitting\n"
      "  kernels into modules based on some heuristic.\n"
      "  The '-split' option is compatible with '-split-esimd'. In this case,\n"
      "  first input module will be split according to the '-split' option\n"
      "  processing algorithm, not distinguishing between SYCL and ESIMD\n"
      "  kernels. Then each resulting module is further split into SYCL and\n"
      "  ESIMD parts if the module has both kinds of entry points.\n"
      "- If -symbols options is also specified, then for each produced module\n"
      "  a text file containing names of all spir kernels in it is generated.\n"
      "- Specialization constant intrinsic transformer. Replaces symbolic\n"
      "  ID-based intrinsics to integer ID-based ones to make them friendly\n"
      "  for the SPIRV translator\n"
      "When the tool splits input module into regular SYCL and ESIMD kernels,\n"
      "it performs a set of specific lowering and transformation passes on\n"
      "ESIMD module, which is enabled by the '-lower-esimd' option. Regular\n"
      "optimization level options are supported, e.g. -O[0|1|2|3|s|z].\n"
      "Normally, the tool generates a number of files and \"file table\"\n"
      "file listing all generated files in a table manner. For example, if\n"
      "the input file 'example.bc' contains two kernels, then the command\n"
      "  $ sycl-post-link --properties --split=kernel --symbols \\\n"
      "    --spec-const=native    -o example.table example.bc\n"
      "will produce 'example.table' file with the following content:\n"
      "  [Code|Properties|Symbols]\n"
      "  example_0.bc|example_0.prop|example_0.sym\n"
      "  example_1.bc|example_1.prop|example_1.sym\n"
      "When only specialization constant processing is needed, the tool can\n"
      "output a single transformed IR file if --ir-output-only is specified:\n"
      "  $ sycl-post-link --ir-output-only --spec-const=emulation \\\n"
      "    -o example_p.bc example.bc\n"
      "will produce single output file example_p.bc suitable for SPIRV\n"
      "translation.\n"
      "--ir-output-only option is not not compatible with split modes other\n"
      "than 'auto'.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;

  if (!DoSplit) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }

  if (IROutputOnly && DoSplit && (SplitMode != SPLIT_AUTO)) {
    errs() << "error: -" << SplitMode.ArgStr << "=" << SplitMode.ValueStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // TODO: check of Output (-o) command line input.

  Expected<StringTable> TableOrErr = processInputModule(std::move(M));
  if (!TableOrErr)
    error(toString(TableOrErr.takeError()));

  auto &Table = *TableOrErr;
  // Input module was processed and a single output file was requested.
  if (IROutputOnly)
    return 0;

  std::error_code EC;
  raw_fd_ostream Out{Output, EC, sys::fs::OF_None};
  checkError(EC, formatv("error opening file: {0}", Output));
  writeStringTable(Table, Out);

  return 0;
}
