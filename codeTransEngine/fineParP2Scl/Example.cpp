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
 "       \\ \nInserted by FinePar    \n"\
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
          if(CallExpr *call= dyn_cast<CallExpr>(ret->getRHS())){
            string  rName= rewriter.getRewrittenText( call->getCallee()->getSourceRange() );
            if(rName=="clSetKernelArg"){
              string  firstElem= rewriter.getRewrittenText( call->getArg(0)->getSourceRange() );
              if(firstElem=="csrKernel"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
                int secondElem=atoi(secondElemStr.c_str());
                if(secondElem>gpuHigh)
                  gpuHigh=secondElem;
              }
              if(firstElem=="csrKernelcpu"){
                string  secondElemStr= rewriter.getRewrittenText( call->getArg(1)->getSourceRange() );
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

    virtual bool VisitFunctionDecl(FunctionDecl *func) {
        numFunctions++;
        string funcName = func->getNameInfo().getName().getAsString();
      func->dump();
        if (funcName == "gpu_csr_ve_slm_pm_fs") {
              int numParas=func->getNumParams();
              auto loc=func->getParamDecl(numParas-1)->getSourceRange().getEnd();
              const char * c = astContext->getSourceManager().getCharacterData(loc);
              int offset = 0;
              while (*c != ')') {
                  ++c;
                    ++offset;
              }
              loc = loc.getLocWithOffset(offset );
              rewriter.InsertText(loc, ", __global unsigned long* bitmap",true,true);
              indicator1=true;
        }
        if (funcName == "cpu_csr") {
              int numParas=func->getNumParams();
              auto loc=func->getParamDecl(numParas-1)->getSourceRange().getEnd();
              const char * c = astContext->getSourceManager().getCharacterData(loc);
              int offset = 0;
              while (*c != ')') {
                  ++c;
                    ++offset;
              }
              loc = loc.getLocWithOffset(offset );
              rewriter.InsertText(loc, ", int rowtotal, __global int* rowall, int rowallsize",true,true);
 
              indicator2=true;
        }
        return true;
    }

    virtual bool VisitStmt(Stmt *st) {

      if(indicator1==true){
      if (ForStmt *ret = dyn_cast<ForStmt>(st)) {
          string tmpStr=rewriter.getRewrittenText( ret->getCond() ->getSourceRange() );
          rewriter.ReplaceText(ret->getCond()->getLocStart(), tmpStr.length(), "row<row_num");
          auto loc=ret->getBody()->getLocStart();
          const char * c = astContext->getSourceManager().getCharacterData(loc);
          int offset = 0;
          while (*c != '{') {
            ++c;
            ++offset;
          }
          loc = loc.getLocWithOffset(offset+1 );

          rewriter.InsertText(loc, "\nif(bitmap[row>>6]&(1ul<<(row&0x3f))){\n",true,true);
          rewriter.InsertText(ret->getSourceRange().getEnd(), "\n}\n",true,true);
          indicator1=false;
      }
      }

      if((indicator1==false) && (indicator2==true)){
        if (ForStmt *ret = dyn_cast<ForStmt>(st)) {
          string tmpStr=rewriter.getRewrittenText( ret->getCond() ->getSourceRange() );
          rewriter.ReplaceText(ret->getCond()->getLocStart(), tmpStr.length(), "r<rowallsize");
          auto loc=ret->getBody()->getLocStart();
          const char * c = astContext->getSourceManager().getCharacterData(loc);
          int offset = 0;
          while (*c != '{') {
            ++c;
            ++offset;
          }
          loc = loc.getLocWithOffset(offset+1 );
          rewriter.InsertText(loc, "\nint row=rowall[r];\n",true,true);
          indicator2=false;
        }
      }

      {
        string tmpStr=rewriter.getRewrittenText( st->getSourceRange() );
        if(tmpStr=="r+=midrow"||tmpStr=="int row=r;")
          rewriter.ReplaceText(st->getLocStart(), tmpStr.length(), "//Deleted");
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
    //int result = Tool.run(newFrontendActionFactory<FrontendActionPreScan>().get());
    int result = Tool.run(newFrontendActionFactory<ExampleFrontendAction>().get());

    errs() << "\nFound " << numFunctions << " functions.\n\n";
    errs()<<"cpuHigh="<<cpuHigh<<"\n";
    errs()<<"gpuHigh="<<gpuHigh<<"\n";

    error_code EC;
    StringRef outPath("./generatedCode/spmv_csr_vector.cl");
    raw_fd_ostream postream(outPath, EC, sys::fs::F_RW);
    if (EC) {
        errs() << EC.message() << "\n";
          exit(0);
    }
    rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(postream);
    return result;
}
