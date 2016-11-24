#include "clang/Driver/Options.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace std;
using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

Rewriter rewriter;
int numFunctions = 0;

bool indicator1;
bool indicator2;
bool indicator3;
int cpuHigh;
int gpuHigh;
string insert2="\n errorCode = clSetKernelArg(csrKernel, 6, sizeof(cl_mem), &devbitmap);\n";
#define insert1 \
 "\n       // Inserted by FinePar    \n"\
 "       unsigned long *bitmap=(unsigned long*)malloc(sizeof(unsigned long)*(((padrowsize)>>6) + 1));               \n " \
 "       int* rowforcpu = (int*)malloc(sizeof(int)*(padrowsize));               \n " \
 "       int rowforcpusum=0;               \n " \
 "       memset(bitmap,0,(sizeof(unsigned long)*(((padrowsize)>>6) + 1)));               \n " \
 "       for(int i=0 ; i<padrowsize; i++){               \n " \
 "         int numi=rowptrpad[i+1]-rowptrpad[i];               \n " \
 "         if(numi<=cpuoffset){               \n " \
 "           bitmap[(i>>6)]=bitmap[(i>>6)]|(1ul<<((i)&0x3f));               \n " \
 "         }               \n " \
 "         else {               \n " \
 "           rowforcpu[rowforcpusum]=i;               \n " \
 "           rowforcpusum++;               \n " \
 "         }               \n " \
 "       }               \n " \
 "       cl_mem devbitmap;               \n " \
 "       ALLOCATE_GPU_READ(devbitmap, bitmap, sizeof(unsigned long)*(((padrowsize)>>6)+1));               \n " \
 "       cl_mem devrowforcpu;               \n " \
 "       ALLOCATE_GPU_READ_cpu(devrowforcpu, rowforcpu, sizeof(int)*(rowforcpusum));               \n\n"





class VisitorPreScan : public RecursiveASTVisitor<VisitorPreScan> {
private:
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit VisitorPreScan(CompilerInstance *CI) 
      : astContext(&(CI->getASTContext())) // initialize private members
    {
        rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
    }

    virtual bool VisitStmt(Stmt *st) {
        if (BinaryOperator *ret = dyn_cast<BinaryOperator>(st)) {
          //errs()<<"-|---> " <<rewriter.getRewrittenText( ret->getRHS()->getSourceRange() )<<"\n";
          if(CallExpr *call= dyn_cast<CallExpr>(ret->getRHS())){
            string  rName= rewriter.getRewrittenText( call->getCallee()->getSourceRange() );
            if(rName=="clSetKernelArg"){
              string  firstElem= rewriter.getRewrittenText( call->getArg(0)->getSourceRange() );
              if(firstElem=="csrKernel"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
                errs()<<"ooooooo->"<<rName<<" "<<firstElem<<" "<<secondElemStr <<"\n";
                int secondElem=atoi(secondElemStr.c_str());
                if(secondElem>gpuHigh)
                  gpuHigh=secondElem;
              }
              if(firstElem=="csrKernelcpu"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
                errs()<<"ooooooo->"<<rName<<" "<<firstElem<<" "<<secondElemStr <<"\n";
                int secondElem=atoi(secondElemStr.c_str());
                if(secondElem>cpuHigh)
                  cpuHigh=secondElem;
              }
            }
          }
        }
        return true;
    }
};

class ExampleVisitor : public RecursiveASTVisitor<ExampleVisitor> {
private:
    ASTContext *astContext; // used for getting additional AST info

public:
    explicit ExampleVisitor(CompilerInstance *CI) 
      : astContext(&(CI->getASTContext())) // initialize private members
    {
        rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
    }


    virtual bool VisitStmt(Stmt *st) {

      if(indicator1==false){
        if (BinaryOperator *ret = dyn_cast<BinaryOperator>(st)) {
          if(CallExpr *call= dyn_cast<CallExpr>(ret->getRHS())){
            string  rName= rewriter.getRewrittenText( call->getCallee()->getSourceRange() );
            if(rName=="clCreateKernel"){
              rewriter.InsertText(ret->getSourceRange().getBegin(), insert1,true,true);
              indicator1=true;
            }
          }
        }
      }
      
      if(indicator2==false){
        if (BinaryOperator *ret = dyn_cast<BinaryOperator>(st)) {
          if(CallExpr *call= dyn_cast<CallExpr>(ret->getRHS())){
            string  rName= rewriter.getRewrittenText( call->getCallee()->getSourceRange() );
            if(rName=="clSetKernelArg"){
              string  firstElem= rewriter.getRewrittenText( call->getArg(0)->getSourceRange() );
              if(firstElem=="csrKernel"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
                int secondElem=atoi(secondElemStr.c_str());
                if(secondElem==gpuHigh){
                  char insertTmp[1000];
                  sprintf(insertTmp, "\nerrorCode = clSetKernelArg(csrKernel, %d, sizeof(cl_mem), &devbitmap);\n",gpuHigh+1);
                  rewriter.InsertText(ret->getLocEnd().getLocWithOffset(2), insertTmp,true,true);
                  indicator2=true;
                }
              }
            }
          }
        }
      }


      if(indicator3==false){
        if (BinaryOperator *ret = dyn_cast<BinaryOperator>(st)) {
          if(CallExpr *call= dyn_cast<CallExpr>(ret->getRHS())){
            string  rName= rewriter.getRewrittenText( call->getCallee()->getSourceRange() );
            if(rName=="clSetKernelArg"){
              string  firstElem= rewriter.getRewrittenText( call->getArg(0)->getSourceRange() );
              if(firstElem=="csrKernelcpu"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
                int secondElem=atoi(secondElemStr.c_str());
                if(secondElem==cpuHigh){
                  errs()<<"ready to insert: \n";
                  char insertTmp[1000];
                  sprintf(insertTmp, "\n   %s %d %s\n    %s %d %s\n    %s %d %s\n ",
                     "errorCode = clSetKernelArg(csrKernelcpu,", cpuHigh+1, ", sizeof(int), &rowforcpu);", 
                     "errorCode = clSetKernelArg(csrKernelcpu,", cpuHigh+2, ", sizeof(cl_mem), &devrowforcpu);", 
                     "errorCode = clSetKernelArg(csrKernelcpu,", cpuHigh+3, ", sizeof(int), &rowforcpusum);"
                     );
                  rewriter.InsertText(ret->getLocEnd().getLocWithOffset(2), insertTmp,true,true);
                  indicator3=true;
                }
              }
            }
          }
        }
      }






        if (ReturnStmt *ret = dyn_cast<ReturnStmt>(st)) {
            errs() << "** Rewrote ReturnStmt\n";
        }        
        if (CallExpr *call = dyn_cast<CallExpr>(st)) {
            errs() << "** Rewrote function call\n";
        }
        return true;
    }

};



class ExampleASTConsumer : public ASTConsumer {
private:
    ExampleVisitor *visitor; // doesn't have to be private

public:
    // override the constructor in order to pass CI
    explicit ExampleASTConsumer(CompilerInstance *CI)
        : visitor(new ExampleVisitor(CI)) // initialize the visitor
    { }

    // override this to call our ExampleVisitor on the entire source file
    virtual void HandleTranslationUnit(ASTContext &Context) {
        /* we can use ASTContext to get the TranslationUnitDecl, which is
             a single Decl that collectively represents the entire source file */
        visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }
};

class ASTConsumerPreScan : public ASTConsumer {
private:
    VisitorPreScan *visitor; // doesn't have to be private

public:
    // override the constructor in order to pass CI
    explicit ASTConsumerPreScan(CompilerInstance *CI)
        : visitor(new VisitorPreScan(CI)) // initialize the visitor
    { }

    // override this to call our VisitorPreScan on the entire source file
    virtual void HandleTranslationUnit(ASTContext &Context) {
        /* we can use ASTContext to get the TranslationUnitDecl, which is
             a single Decl that collectively represents the entire source file */
        visitor->TraverseDecl(Context.getTranslationUnitDecl());
    }
};


class ExampleFrontendAction : public ASTFrontendAction {
public:
    virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) {
        return std::unique_ptr<clang::ASTConsumer> (new  ExampleASTConsumer(&CI) ); // pass CI pointer to ASTConsumer
    }
};

class FrontendActionPreScan : public ASTFrontendAction {
public:
    virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) {
        return std::unique_ptr<clang::ASTConsumer> (new  ASTConsumerPreScan(&CI) ); // pass CI pointer to ASTConsumer
    }
};

 static llvm::cl::OptionCategory MyToolCategory("my-tool options");//zf

int main(int argc, const char **argv) {
  indicator1=false;
  indicator2=false;
  indicator3=false;
gpuHigh=0;
cpuHigh=0;
    // parse the command-line args passed to your code
    CommonOptionsParser op(argc, argv, MyToolCategory);        
    // create a new Clang Tool instance (a LibTooling environment)
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    // run the Clang Tool, creating a new FrontendAction (explained below)
    int result = Tool.run(newFrontendActionFactory<FrontendActionPreScan>().get());
//    errs()<<"cpuHigh: "<<cpuHigh<<"\n"; exit(0);
    result = Tool.run(newFrontendActionFactory<ExampleFrontendAction>().get());

    errs() << "\nFound " << numFunctions << " functions.\n\n";
    errs()<<"cpuHigh="<<cpuHigh<<"\n";
    errs()<<"gpuHigh="<<gpuHigh<<"\n";
    // print out the rewritten source code ("rewriter" is a global var.)
    error_code EC;
    StringRef outPath("./generatedCode/spmv_csr.cpp");
    raw_fd_ostream postream(outPath, EC, sys::fs::F_RW);
    if (EC) {
      errs() << EC.message() << "\n";
      exit(0);
    }
    rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(postream);
    return result;
}
