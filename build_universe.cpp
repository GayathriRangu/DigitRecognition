#include<stdio.h>
#include<iostream>
//#include<cstdio>
#include<fstream>
#include<vector>
#include <algorithm>
#include<string>
#include<sstream>
#include<cmath>
#include<math.h>


using namespace std;

#define SILENCE 200000
#define F 4001
#define K 33
#define ci 13
#define N 6


int main(){
	ofstream fout_universe_all;
	ifstream fin_digit;
	std::string amp_string;
	std::string universe_file_str="input\\Universe_S.txt";
	std::string input_file_old="input\\";
	std::string input_file;
	char* digit_str_array[10]={"zero","one","two","three","four","five","six","seven","eight","nine"};
	fout_universe_all.open(universe_file_str.c_str(),ios::out|ios::trunc);	
	//fout_universe_all.open(universe_file_str.c_str(),ios::out|ios::app);

	if(fout_universe_all.is_open()){
	for(int m=0;m<10;m++){
		cout<<"Processing "<<digit_str_array[m]<<endl;
		for(int n=1;n<=15;n++){
			input_file=input_file_old+digit_str_array[m]+"\\"+digit_str_array[m]+"_"+to_string(static_cast<long long>(n))+".txt";
			fin_digit.open(input_file.c_str());
			if(fin_digit.is_open()){
				while(getline(fin_digit,amp_string)){
			
					fout_universe_all<<amp_string<<endl;
			
				}
			}//end of if
			fin_digit.close();
		}
	}
	fout_universe_all.close();
	}
	else{

		cout<<"universe file is not open"<<endl;
	}
	
}