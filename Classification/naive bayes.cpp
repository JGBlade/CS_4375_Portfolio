#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

/*
Naive Bayes
*/
using namespace std;

//Calculates the sum of the given vector
double sumVect(vector<double> input)
{
    double total = 0;

    for(int i = 0; i < (int)input.size(); i++)
    {
        total += input[i];
    }

    return total;
}

//Calculates the mean of the given vector
double meanVect(vector<double> input)
{
    double mean = 0;

    mean = (sumVect(input) / input.size());

    return mean;
}

//Calculate the variance of a given vector
double varVect(vector<double> input)
{
    double mean = 0, var = 0;

    mean = meanVect(input);

    for(int i = 0; i < (int)input.size(); i++)
    {
        var += pow((double)(input[i] - mean), 2.0);
    }

    var = (double)var / (double)input.size();

    return var;
}

//Calculate the likelihood of an outcome given the parameters. This function can only be used for qualitative predictors
void likelihood(vector<int> data, int factor, vector<int> survived, int survCount, int dieCount, double &likeSurv, double &likeDie)
{
    int classSurv = 0, classDie = 0;

    for(int i = 0; i < (int)data.size(); i++)
    {
        if(data[i] == factor)
        {
            if(survived[i] == 1)
                classSurv++;
            else
                classDie++;
        }
    }

    likeSurv = (double)((double)classDie / (double)dieCount);
    likeDie = (double)((double)classSurv / (double)survCount);

}

//This function calculates the likelihood of each predictor the returns the prior probability of survival
double naiveBayes(vector<int> survived, vector<int> pClass, vector<int> sex, vector<double> age, double (&likePclass)[3][2], double (&likeSex)[2][2], double (&probAge)[2])
{
    double probSurvive, tempLikeSurv, tempLikeDie;
    int survCount = 0, dieCount = 0;

    for(int i = 0; i < (int)survived.size(); i++)
    {
        if(survived[i] == 1)
            survCount++;
        else
            dieCount++;
    }

    //Calculate priors
    probSurvive = (double)((double)survCount / (double)survived.size());

    //Calculate likelihood for each predictor, except age, that will be calculated later
    likelihood(pClass, 1, survived, survCount, dieCount, tempLikeSurv, tempLikeDie);
    likePclass[0][0] = tempLikeDie;
    likePclass[0][1] = tempLikeSurv;

    likelihood(pClass, 2, survived, survCount, dieCount, tempLikeSurv, tempLikeDie);
    likePclass[1][0] = tempLikeDie;
    likePclass[1][1] = tempLikeSurv;

    likelihood(pClass, 3, survived, survCount, dieCount,  tempLikeSurv, tempLikeDie);
    likePclass[2][0] = tempLikeDie;
    likePclass[2][1] = tempLikeSurv;

    likelihood(sex, 0, survived, survCount, dieCount,  tempLikeSurv, tempLikeDie);
    likeSex[0][0] = tempLikeDie;
    likeSex[0][1] = tempLikeSurv;

    likelihood(sex, 1, survived, survCount, dieCount,  tempLikeSurv, tempLikeDie);
    likeSex[1][0] = tempLikeDie;
    likeSex[1][1] = tempLikeSurv;

    return probSurvive;
}

//This function calculates the likelihood of survival for a given age,
double calcProbAge(double age, double mean, double var)
{
    double prob ((1.0 / sqrt(2.0 * M_PI * var)) * exp( -(pow((age - mean), 2.0)) / (2 * var)));

    return prob;
}

//Calculates the raw probability for survival and death for each of the test instances
void rawProb(double (&results)[246][2], vector<int> pClass, vector<int> sex, vector<double> age, double (&likePclass)[3][2], double (&likeSex)[2][2], double probSurv, double (&meanAge)[2], double (&varAge)[2])
{
    double num_s, num_d, denom;
    double probDie = 1 - probSurv;

    //Loop through each test instance and calculate raw probabilities
    for(int i = 0; i < (int)pClass.size(); i++)
    {
        num_s = (double)likePclass[pClass[i]][1] * (double)likeSex[sex[i]][1] * probSurv * (calcProbAge(age[i], meanAge[1], varAge[1]));    //Survival numerator
        num_d = (double)likePclass[pClass[i]][0] * (double)likeSex[sex[i]][0] * probDie * (calcProbAge(age[i], meanAge[0], varAge[0]));     //Death numerator

        //denominator
        denom = (double)likePclass[pClass[i]][1] * (double)likeSex[sex[i]][1] * probSurv * (calcProbAge(age[i], meanAge[1], varAge[1])) + (double)likePclass[pClass[i]][0] * (double)likeSex[sex[i]][0] * probDie * (calcProbAge(age[i], meanAge[0], varAge[0]));

        results[i][0] = (num_d / denom);
        results[i][1] = (num_s / denom);
    }

}

//Calculates accuracy given confusion matrix values
double accuracy(int truePos, int trueNeg, int fNeg, int fPos)
{
    return (double)(trueNeg + truePos)/(trueNeg + truePos + fNeg + fPos);
}

//Calculates sensitivity given confusion matrix values
double sensitivity(int truePos, int fNeg)
{
    return (double)(truePos)/(truePos + fNeg);
}

//Calculates specificity given confusion matrix values
double specificity(int trueNeg, int fPos)
{
    return (double)(trueNeg)/(trueNeg + fPos);
}

int main(int argc, char** argv)
{
    ifstream inFile;
    string input;
    int numObvs = 0;
    string whoCares, pClassIn, survivedIn, sexIn, ageIn;

    vector<int> pClass(1048);
    vector<int> survived(1048);
    vector<int> sex(1048);
    vector<double> age(1048);

    inFile.open("titanic_project.csv");

    if(!inFile.is_open())
    {
        cout << "The file was not opened properly." << endl;
        return 1;
    }

    cout << "File loaded successfully." << endl << endl;

    getline(inFile, input);
    cout << "Heading: " << input << endl;

    while(inFile.good())
    {
        getline(inFile, whoCares, ',');
        getline(inFile, pClassIn, ',');
        getline(inFile, survivedIn, ',');
        getline(inFile, sexIn, ',');
        getline(inFile, ageIn, '\n');

        pClass.at(numObvs) = stoi(pClassIn);
        survived.at(numObvs) = stoi(survivedIn);
        sex.at(numObvs) = stoi(sexIn);
        age.at(numObvs) = stof(ageIn);

        numObvs++;
    }

    pClass.resize(numObvs);
    survived.resize(numObvs);
    sex.resize(numObvs);
    age.resize(numObvs);

    inFile.close();

    vector<int> pClassTrain(800);
    vector<int> survivedTrain(800);
    vector<int> sexTrain(800);
    vector<double> ageTrain(800);

    vector<int> pClassTest(300);
    vector<int> survivedTest(300);
    vector<int> sexTest(300);
    vector<double> ageTest(300);

    //Here we'll split the data into train and test
    for(int i = 0; i < (int)pClass.size(); i++)
    {
        if(i < 800)
        {
            pClassTrain[i] = pClass[i];
            survivedTrain[i] =  survived[i];
            sexTrain[i] = sex[i];
            ageTrain[i] = age[i];
        }
        else
        {
            pClassTest.at(i - 800) = pClass[i];
            survivedTest.at(i - 800) =  survived[i];
            sexTest.at(i - 800) = sex[i];
            ageTest.at(i - 800) = ageTrain[i];
        }
    }

    pClassTest.resize(pClass.size() - 800);
    survivedTest.resize(survived.size() - 800);
    sexTest.resize(sex.size() - 800);
    ageTest.resize(age.size() - 800);

    double likePclass[3][2];        //Likelihood for passenger class survival stored in 2D array where [class - 1][survival]
    double likeSex[2][2];           //Likelihood for survival based on sex stored where in 2D array where [sex][survival]
    double probAge[2];

    cout << "Begin Naive Bayes..." << endl;

    //start timer
    auto start = chrono::steady_clock::now();

    double probSurv = naiveBayes(survivedTrain, pClassTrain, sexTrain, ageTrain, likePclass, likeSex, probAge);

    //end timer
    auto end = chrono::steady_clock::now();
    cout << "Model completed in " << chrono::duration_cast<chrono::nanoseconds>(end - start).count() << " nano seconds. \n" << endl;

    //count up survivors
    int survCount = 0;
    for(int i = 0; i < (int)survivedTrain.size(); i++)
    {
        if(survived[i] == 1)
            survCount++;
    }

    int dieCount = 800 - survCount;

    vector<double> ageSurv(survCount);
    vector<double> ageDie(dieCount);

    int counterLive = 0, counterDead = 0;

    //Split age into vectors based on survival
    for(int i = 0; i < (int)survivedTrain.size(); i++)
    {
        if(survived[i] == 1)
        {
            ageSurv[counterLive] = ageTrain[i];
            counterLive++;
        }
        else
        {
            ageDie[counterDead] = ageTrain[i];
            counterDead++;
        }
    }

    //Calculate values of mean and variance based on survival
    double ageMeanSurv = meanVect(ageSurv);
    double ageVarSurv = varVect(ageSurv);      //Sigma squared

    double ageMeanDie = meanVect(ageDie);
    double ageVarDie = varVect(ageDie);        //Sigma squared

    double ageMean[2] = {ageMeanDie, ageMeanSurv};
    double ageVar[2] = {ageVarDie, ageVarSurv};

    double results[246][2];

    //Get raw probabilities for each instance of test
    rawProb(results, pClassTest, sexTest, ageTest, likePclass, likeSex, probSurv, ageMean, ageVar);

    cout << "Probabilities for First 5 Instances: " << endl;
    cout << "0 \t 1" << endl;
    cout << results[0][0] << " " << results[0][1] << endl;
    cout << results[1][0] << " " << results[1][1] << endl;
    cout << results[2][0] << " " << results[2][1] << endl;
    cout << results[3][0] << " " << results[3][1] << endl;
    cout << results[4][0] << " " << results[4][1] << endl << endl;

    int predictions[246];

    //Make predictions based on generated probabilities
    for(int i = 0; i < 246; i++)
    {
        if(results[i][1] > results[i][0])
            predictions[i] = 0;
        else
            predictions[i] = 1;
    }

    int truePos = 0, trueNeg = 0, fPos = 0, fNeg = 0;

    //Calculate values for confusion matrix
    for(int i = 0; i < (int)survivedTest.size(); i++)
    {
        if(predictions[i] == survivedTest[i])
            {
                if((predictions[i] == 1) && (survivedTest[i] == 1))
                    truePos++;
                else
                    trueNeg++;
            }
            else
                if(predictions[i] == 1)
                    fPos++;
                else
                    fNeg++;
    }

    cout << "Confusion Matrix: " << endl;
    cout << "\t T\t F" << endl;
    cout << "T: \t" << truePos << "\t" << fPos << endl;
    cout << "F: \t" << fNeg << "\t" << trueNeg << endl;

    cout << endl << "Accuracy: " << accuracy(truePos, trueNeg, fNeg, fPos) << endl;
    cout << "Sensitivity: " << sensitivity(truePos, fNeg) << endl;
    cout << "Specificity: " << specificity(trueNeg, fPos) << endl;
    return 0;
}
