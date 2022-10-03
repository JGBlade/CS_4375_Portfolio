#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;

//This is a simple implementation of the sigmoid function, defined as 1 / (1 + e^-x)
double sigmoidFunc(double x)
{
    return (1.0 / (1.0 + exp(-x)));
}

//Function for logistic regression, returns odds for each instance in a corresponding vector
vector<double> logRegress(vector<int> sex, vector<int> survived)
{
    double learningRate = 0.001, probTemp;
    vector<double> weights(2);
    double dataMatrix[2][800];
    vector<double> prob(800);
    vector<double> labels(800);
    vector<double> error(800);
    vector<double> errorCof(2);

    weights[0] = 1;
    weights[1] = 1;

    //Create data matrix with first column set to one
    for(int i = 0; i < (int)sex.size(); i++)
    {
        dataMatrix[0][i] = 1;
        dataMatrix[1][i] = sex[i];
    }


    for(int i = 0; i < (int)survived.size(); i++)
    {
        labels[i] = survived[i];
    }


    cout << "Begin training iterations..." << endl;

    //Start clock
    auto start = chrono::steady_clock::now();
    for(int i = 0; i < 500000; i++)
    {
        for(int j = 0; j < 800;  j++)
        {
            probTemp = (dataMatrix[0][j] * weights[0]) + (dataMatrix[1][j] * weights[1]);
            prob[j] = sigmoidFunc(probTemp);
        }

        for(int j = 0; j < (int)prob.size(); j++)
        {
            error[j] = labels[j] - prob[j];
        }

        for(int j = 0; j < (int)errorCof.size(); j++)
        {
            for(int k = 0; k < (int)error.size(); k++)
            {
                errorCof[j] += (dataMatrix[j][k] * error[k]);
            }
        }

        weights[0] = weights[0] + (learningRate * errorCof[0]);
        weights[1] = weights[1] + (learningRate * errorCof[1]);

    }

    //stop clock
    auto end = chrono::steady_clock::now();

    cout << "Training iterations finished in " << chrono::duration_cast<chrono::seconds>(end - start).count() << " seconds. \n" << endl;
    return weights;
}

//Calculate accuracy based on confusion matrix values
double accuracy(int truePos, int trueNeg, int fNeg, int fPos)
{
    return (double)(trueNeg + truePos)/(trueNeg + truePos + fNeg + fPos);
}

//calculates sensitivity based on confusion matrix values
double sensitivity(int truePos, int fNeg)
{
    return (double)(truePos)/(truePos + fNeg);
}

//calculates specificity based on confusion matrix values
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

    //We'll collect values for each predictor but we'll only use sex to predict survival
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

    //Split the data into train and test
    for(int i = 0; i < (int)pClass.size(); i++)
    {
        if(i < 800)
        {
            pClassTrain[i] = pClass[i];
            survivedTrain[i] =  survived[i];
            sexTrain[i] = sex[i];
            ageTrain[i] = ageTrain[i];
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

    //Get coefficients for model
    vector<double> trainWeights = logRegress(sexTrain,survivedTrain);

    cout << "Coefficients: \n" << trainWeights[0] << endl << trainWeights[1] << endl << endl;

    double testMatrix[2][survivedTest.size()];

    //Set the values for the test matrix
    for(int i = 0; i < (int)survivedTest.size(); i++)
    {
        testMatrix[0][i] = 1;
        testMatrix[1][i] = sexTest[i];
    }

    vector<double> predTest(246);

    //Make predictions using the model
    for(int i = 0; i < (int)predTest.size(); i++)
    {
        predTest[i] = (testMatrix[0][i] * trainWeights[0]) + (testMatrix[1][i] * trainWeights[1]);
    }

    vector<double> probs(246);

    //generate probability from log odds
    for(int i = 0; i < (int)predTest.size(); i++)
    {
        probs[i] = (exp(predTest[i]) / (1 + exp(predTest[i])));
    }

    vector<int> finalPred(246);

    //If the value for an instance is greater than 0.5, predict survival
    for(int i =0; i < (int)probs.size(); i++)
    {
        if(probs[i] >= 0.5)
            finalPred[i] = 1;
        else
            finalPred[i] = 0;
    }

    int truePos = 0, trueNeg = 0, fPos = 0, fNeg = 0;

    //calculate values for confusion matrix
    for(int i = 0; i < (int)finalPred.size(); i++)
    {
        if(finalPred[i] == survivedTest[i])
        {
            if((finalPred[i] == 1) && (survivedTest[i] == 1))
                truePos++;
            else
                trueNeg++;
        }
        else
            if(finalPred[i] == 1)
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
