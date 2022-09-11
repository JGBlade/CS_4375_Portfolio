#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

/********************************************************************\
Jonathan Blade
Portfolio Component 1: Data Exploration

This program contains several functions used to explore data stored
in vectors including, sum, mean, median, range, covariance, and
correlation. This program is designed to be used with a particular
data file, "Boston.csv", however the functions can work independently
of that file.
\********************************************************************/




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

//Sorts the elements of a vector, a helper function to be used elsewhere
vector<double> sortVect(vector<double> input)
{
    double holder = 0;

    vector<double> temp = input;

    //Bubble Sort
    for(int i = 0; i < (int)temp.size(); i++)
    {
        for(int j = 0; j < (int)(temp.size() - i - 1); j++)
        {
            if(temp[j] > temp[j + 1])
            {
                holder = temp[j];
                temp[j] = temp[j + 1];
                temp[j + 1] = holder;
            }
        }
    }

    return temp;
}

//Finds the median of the given vector
double medianVect(vector<double> input)
{
    double median = 0;
    vector<double> temp = input;

    //Sort the vector
    temp = sortVect(temp);

    //If vector has an even number of elements
    if(temp.size() % 2 == 0)
    {
        double mid, nextMid;
        mid = temp[temp.size()/2];
        nextMid = temp[(temp.size()/2) + 1];

        median = (mid + nextMid) / 2;   //calculate average of two middle elements.
    }
    else
        median = temp[temp.size()/2]; //Otherwise, get the middle element


    return median;
}

//Finds the range of the given vector
void rangeVect(vector<double> input, double& lower, double& upper)
{
    vector<double> temp = input;

    //Sort vector
    temp = sortVect(temp);

    //Lower and upper are passed by reference, not returned
    lower = temp[0];
    upper = temp[temp.size() - 1];
}

//Calculate the covariance of the given vectors
double covar(vector<double> rm, vector<double> medv)
{
    vector<double> tempRm = rm, tempMedv = medv;
    double numeratorSum = 0, result = 0;

    //the numerator of the formula is calculated
    for(int i = 0; i < (int)tempRm.size(); i++)
    {
        numeratorSum += (tempRm[i] - meanVect(tempRm)) * (tempMedv[i] - meanVect(tempMedv)); //The sum of the element Xi minus X-Bar multiplied by the element Yi minus Y-Bar.
    }

    //Divide the numerator by the number of elements minus 1
    result = (numeratorSum / (double)(tempRm.size() - 1));

    return result;
}

//Calculates the standard variation of the given vector
//to be used in calculating the correlation
double sDevVect(vector<double> input)
{
    vector<double> temp = input;
    double result = 0, sum = 0, mean = 0;

    //Calculate the mean
    mean =  meanVect(temp);

    //Sum the square minus the mean of each element
    for(int i = 0; i < (int)temp.size(); i++)
    {
        sum += pow(temp[i] - mean, 2);
    }

    //Calculate the square root of the sum divided by the number of elements
    result = sqrt(sum / (double)temp.size());

    return result;
}

//Calculates the correlation of the given vectors
double cor(vector<double> rm, vector<double> medv)
{
    vector<double> tempRm = rm, tempMedv = medv;
    double result = 0;

    //calculate the covariance divided by the product of the standard deviation of both vectors
    result = covar(tempRm,tempMedv) / (sDevVect(tempRm) * sDevVect(tempMedv));

    return result;
}

int main(int argc, char** argv)
{
    ifstream inFile;
    string input;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);
    double lower = 0, upper = 0;

    inFile.open("Boston.csv"); //Open data file

    if(!inFile.is_open())
    {
        cout << "Mission Failed." << endl;
        return 1;
    }

    cout << "File loaded successfully." << endl << endl;

    //First line is the heading
    getline(inFile, input);
    cout << "Heading: " << input << endl;

    int numObvs = 0;

    while(inFile.good())
    {
        //Get the data for each vector
        getline(inFile, rm_in, ',');
        getline(inFile, medv_in, '\n');

        //Store the data in the corresponding vector
        rm.at(numObvs) = stof(rm_in);
        medv.at(numObvs) = stof(medv_in);

        numObvs++; //Keep track of how many rows
    }

    inFile.close(); //Close the file

    //Resize the vectors
    rm.resize(numObvs);
    medv.resize(numObvs);

    cout << "New Vector Length: " << rm.size() << endl;
    cout << "Number of Records: " << numObvs << endl;

    cout << "\nStats for RM: " << endl;
    cout << "Sum: " << sumVect(rm) << endl;
    cout << "Mean: " << meanVect(rm) << endl;
    cout << "Median: " << medianVect(rm) << endl;
    rangeVect(rm, lower, upper);
    cout << "Range: " << (upper - lower) << " (" << lower << " to " << upper << ")" << endl;

    cout << "\nStats for MedV: " << endl;
    cout << "Sum: " << sumVect(medv) << endl;
    cout << "Mean: " << meanVect(medv) << endl;
    cout << "Median: " << medianVect(medv) << endl;
    rangeVect(medv, lower, upper);
    cout << "Range: " << (upper - lower) << " (" << lower << " to " << upper << ")" << endl;

    cout << "\nCovariance for RM and MedV: " << covar(rm,medv) << endl;

    cout << "\nCorrelation for RM and MedV: " << cor(rm,medv) << endl;

    cout << "\nWe're done here." << endl;
    return 0;
}
