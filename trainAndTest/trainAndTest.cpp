#include "generial.h"
#include "svm.h"

#define INF HUGE_VAL
static int (*info)(const char *fmt,...) = &printf;

struct svm_parameter param;     // set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_model* model;
struct svm_node *x_space;
struct svm_node *x;

using namespace std;

void saveData(int ** Data,char *filename,int frames, const int dim)
{
    ofstream file(filename);

    for (int i = 0; i < frames; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            file<<Data[i][j]<<" ";
        }
        file<<endl;
    }
    file.close();
}


void readDarWinfromFile(char * fvFilePath, float **data, int index)
{
    ifstream file(fvFilePath);
    int flag = 0;
    for (int i = 0; i < 2*DIMENSION; ++i)
    {
       file >> data[index][i];
       if(data[index][i] != 0)
       {
        flag = 1 ? data[index][i] > 0 : -1;
        data[index][i] =flag * sqrt(fabs(data[index][i]));
       }
    }
    file.close();
    return ;
}
void darWinNormalizedL2(float ** Data, int frames, int di)
{
    float sum = 0;
    for (int i = 0; i < frames; ++i)
    {
        sum = 0;
        for (int j = 0; j < di; ++j)
        {
            sum += fabs(Data[i][j]) *fabs(Data[i][j]);
        }
        sum = sqrt(sum);
        for (int j = 0; j < di; ++j)
        {
            Data[i][j] /= sum;
        }
    }
    return;
}
void initiateParam()
{
    
// default values
    param.svm_type = C_SVC;
    param.kernel_type = PRECOMPUTED;
    param.degree = 3;
    param.gamma = 1.0/(trainNum+1);  
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;  // a parameter decided by cross-validation
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
}
double CrossValidation(int nFolds)
{
    double best_C = 1;
    int maxCorrect = 0;
    double *target = Malloc(double, prob.l);
    
    // for each candidate C, using cross validation
    for (int iChoice = 0; iChoice < cChoice; iChoice++)
    {
        initiateParam();
        param.C = pow(2, C[iChoice]);
        svm_cross_validation(&prob, &param, nFolds, target);
        int nCorrect = 0;
        for (int iTrain = 0; iTrain < prob.l; iTrain++)
        {
            if (target[iTrain] == prob.y[iTrain])
            {
                nCorrect++;
            }
        }
        if (nCorrect > maxCorrect)
        {
            maxCorrect = nCorrect;
            best_C = param.C;
        }
    }
    delete target;
    return best_C;
}
void trainAndClassify(float **trainData,int Dimen,int **classlabel, int trainNum, float **testData, int testNum)
{
    double  *accuracy = new double[actionType];
    for (int k = 0; k < actionType ; ++k)
    {
        cout<<k<<" ==============================="<<endl;
        //read_problem
        prob.l = trainNum;
        size_t elements = trainNum*(Dimen+1);
        
        int  i,j,s ;
        prob.y = Malloc(double,prob.l);
        prob.x = Malloc(struct svm_node *,prob.l);
        x_space = Malloc(struct svm_node,elements);

        j = 0;
        for(i=0;i<prob.l;i++)
        {
            prob.y[i] = classlabel[i][k];
            prob.x[i] = &x_space[j];
            for (int s = 0; s < Dimen; ++s)
            {
                x_space[j].index = s;
                //cout<<trainData[i][s]<<" ";
                x_space[j++].value = trainData[i][s];
            }
            //cout<<endl;
            x_space[j++].index = -1;
        }
        
        print_func = &print_null;
        svm_set_print_string_function(print_func);
        double best_C = CrossValidation(5); // five-fold cross validation.
        initiateParam();
        param.C = best_C;
        //cout<<" before training "<<endl;
        model = svm_train(&prob,&param);
        //cout<<" after training "<<endl;
        int correct = 0;
        int predict_label;
        int total = testNum;
        x = (struct svm_node *) malloc(Dimen*sizeof(struct svm_node));
        for(i=0;i<testNum;i++)
        {
            for ( s = 0; s < Dimen; ++s)
            {
                x[s].index = s;
                x[s].value = testData[i][s];
            }
            x[s].index = -1;
            predict_label = svm_predict(model,x);
            if(predict_label == classlabel[i+trainNum][k]) //need to change ............===============================
                ++correct;
        }
        svm_free_and_destroy_model(&model);
        svm_destroy_param(&param);
        free(prob.y);
        free(prob.x);
        free(x_space);
        free(x);
        double tempaccuracy = (double)correct/total*100;
        accuracy[k] = tempaccuracy;
        //cout<<accuracy[k]<<" is accuracy for action "<<k<<endl;
    }
    ofstream file(accuracyFile);
    double sum = 0;
    for (int i = 0; i < actionType; ++i)
    {
        sum += accuracy[i];
        file << accuracy[i]<<" ";
    }
    file << endl << " The average accuracy is ";
    file << sum/actionType << endl;
    file.close();
    delete []accuracy;
}
void ReadtrainAndTestFilePath(float ** trainData, float ** testData)
{
    ifstream file(trainAndTestFilePath);
    for (int i = 0; i < trainNum; ++i)
    {
        for (int j = 1; j <= trainNum ; ++j)
        {
            file >> trainData[i][j];
        }
    }
    for (int i = 0; i < testNum; ++i)
    {
        for (int j = 1; j <= trainNum ; ++j)
        {
            file >> testData[i][j];
        }
    }
    file.close();
}
int main(int argc, char const *argv[])
{
    char **fullvideoname = getFullVideoName();

    max_iline_len = 1024;
    iline = Malloc(char,max_iline_len);
    
    int ** classid = readLabelFromFile();
    //saveData(classid,"./classid",datasetSize,actionType);
   
    float ** trainData = new float*[trainNum];
    for (int i = 0; i < trainNum; ++i)
    {
        trainData[i] = new float[trainNum+1];
        trainData[i][0] = i+1;
    }
    float ** testData = new float*[testNum];
    for (int i = 0; i < testNum; ++i)
    {
        testData[i] = new float[trainNum+1];
        testData[i][0] = i+1;
    }
    ReadtrainAndTestFilePath(trainData,testData);

    darWinNormalizedL2(trainData, trainNum, trainNum+1);
    darWinNormalizedL2(testData, testNum, trainNum+1);


    trainAndClassify(trainData,trainNum+1,classid,trainNum, testData, testNum);

    releaseFullVideoName(fullvideoname);

    for (int i = 0; i < trainNum; ++i)
    {
        delete [] trainData[i];
    }
    delete []trainData;

    for (int i = 0; i < trainNum; ++i)
    {
        delete [] testData[i];
    }
    delete []testData;
    free(iline);
    return 0;
}


