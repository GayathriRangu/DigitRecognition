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
#define F 8001
#define K 33
#define ci 13
#define N 6

std::string input_file="test\\nine_34.txt";
std::string centroids_file_str="logs\\Universe\\Centroids.txt";

std::string a_star_file_str="test\\a_star.txt";
std::string b_star_file_str="test\\b_star.txt";
std::string pi_star_file_str="test\\pi_star.txt";
std::string p_star_file_str="test\\p_star.txt";
std::string q_star_file_str="test\\q_star.txt";

std::string file_old="logs\\";
std::string input_file_old="test\\";

vector<int> samples_digit;
long double max_score;
int no_of_samples;
int no_of_frames;
int new_no_of_frames;
long double yis[K][ci],xis[F][ci+1];
long double weights[ci];
int obs_seq[8000];
long double a[N][N],b[N][K],pi_mat[N],a_avg[N][N],b_avg[N][K],pi_avg[N];
int psi[F][N],q_star[F];
long double alpha[F][N],beta[F][N],gamma[F][N],p_star,delta[F][N],zeta[F][N][N],p_star_old;
std::string curr_digit,max_score_digit;



std::string norm_file;
std::string cval_file_str;
std::string frame_skip_str;

std::string frames_vectors;

std::string a_file_str="input\\a.txt";
std::string b_file_str="input\\b.txt";
std::string pi_file_str="input\\pi.txt";


std::string a_avg_file_str;
std::string b_avg_file_str;

void normalise();
bool ci_val_func(int,int);
long double calculate_rval(int ,int ,int );
void initialise_weights();
void create_obs_sequence();
void read_a_b_pi();

void train_HMM();
long double forward_procedure();
void backward_procedure();
void solution_two();
long double viterbi_algo();
void solution_three();

int main(){
	int i,j,count=0,k;
	char* digit_str_array[10]={"zero","one","two","three","four","five","six","seven","eight","nine"};
	
	/*for(int m=0;m<2;m++){		
		a_file_str=file_old+digit_str_array[m]+"\\a_avg.txt";
		b_file_str=file_old+digit_str_array[m]+"\\b_avg.txt";
		pi_file_str=file_old+digit_str_array[m]+"\\pi.txt";
		
		cout<<digit_str_array[m]<<" score"<<endl;*/
	
	samples_digit.erase(samples_digit.begin(),samples_digit.end());
	 no_of_samples=0;
	 no_of_frames=0;
	 new_no_of_frames=0;
	 		 max_score=99.0e-400;
	 for(int o=0;o<10;o++){		
		 samples_digit.erase(samples_digit.begin(),samples_digit.end());

	 no_of_samples=0;
	 no_of_frames=0;
	 new_no_of_frames=0;
		a_file_str=file_old+digit_str_array[o]+"\\a_avg.txt";
		b_file_str=file_old+digit_str_array[o]+"\\b_avg.txt";
		pi_file_str=file_old+digit_str_array[o]+"\\pi.txt";
		
		cout<<digit_str_array[o]<<" score ";
		curr_digit=digit_str_array[o];
	 //cout<<"Processing "<<digit_str_array[m]<<" "<<n<<"utterance"<<endl;
	
	norm_file=input_file_old+"\\norm.txt";
	cval_file_str=input_file_old+"\\cval.txt";
	frame_skip_str=input_file_old+"\\frame_skip.txt";
	frames_vectors=input_file_old+"\\xis_clusters.txt";	
	
	bool skip_frame;
	
	std::string ci_string;
		
	normalise();
	
	ofstream fout_ci_val;
	fout_ci_val.open(cval_file_str.c_str(),ios::out|ios::trunc);
	fout_ci_val.close();
	
	ofstream fout_frame_skip;
	fout_frame_skip.open(frame_skip_str.c_str(),ios::out|ios::trunc);
	
	no_of_frames=(((int)(no_of_samples/320))*4)-4;
	//cout<<"The number of frames is "<<no_of_frames<<endl;
		
	for(i=0,j=1;i<no_of_frames;i++,j++){
		skip_frame=ci_val_func(i,j);
		if(skip_frame){
			//cout<<"The frame skipped is "<<i<<endl;
			fout_frame_skip<<i<<endl;
			j--;count++;
			//cout<<"count is "<<count<<endl;
		}
	}//end of for
	fout_frame_skip.close();
	//cout<<input_file<<endl;
	//cout<<"The no of frames after skipping is "<<j-1<<endl;
	new_no_of_frames=j-1;
	initialise_weights();
	
	create_obs_sequence();
	
	read_a_b_pi();
	
	train_HMM();
	

	
	}//end of for loop
	
	cout<<"The digit recognised is "<<max_score_digit<<" with the score of "<<max_score<<endl;

}//end of main


void normalise(){
	//cout<<"In normalise function"<<endl;
	//cout<<norm_file<<endl;
	ifstream fin_digit;
	std::string amp_string;
	ofstream fout_norm_digit;
	long int max,temp;
	int i;
	
	fin_digit.open(input_file.c_str());
	
	if(fin_digit.is_open()){
	
		fout_norm_digit.open(norm_file.c_str(),ios::out|ios::trunc);
	
		getline(fin_digit,amp_string);
		max=(atoi(amp_string.c_str()));
		samples_digit.push_back(max);
		max=abs(samples_digit[0]);
	
		for(i=1;getline(fin_digit,amp_string) ;i++){
			temp=(atoi(amp_string.c_str()));
			samples_digit.push_back(temp);
			
			if(max<abs(samples_digit[i]))
				max=abs(samples_digit[i]);
				
		}
		
		no_of_samples=i;
		//cout<<"the number of samples is "<<no_of_samples<<endl;
		
		for(i=0;i<no_of_samples;i++){
			temp=0;
			temp=(5000*samples_digit[i])/max;
			samples_digit[i]=temp;
			fout_norm_digit<<samples_digit[i]<<endl;
		}
	
		fin_digit.close();
		fout_norm_digit.close();
		
		return;

	}//end of if
	
	else{
	
		cout<<"The file is not open"<<endl;
		exit(1);
	
	}//end of else

}//end of function normalise

bool ci_val_func(int start,int frame_no){

	int first,last,i,j,m,s;
	long double win,hamm=0.0,energy=0.0;
	long double r[13],a[13],inval;
	long double c[13],am1[13],km,em1,em;

	first=80*start;
	last=first+319;
	
	for(i=first;i<=last;i++){
	
		energy+=samples_digit[i]*samples_digit[i];
	
	}//end of energy for
	
	if(energy<=SILENCE){
		//cout<<"The energy is "<<energy<<endl;
		return true;
	}
	
	for(i=first,j=0;i<=last;i++,j++){
	
		win=0.54-0.46*cos((2*3.142*(j))/319);
		hamm=win*samples_digit[i];
		samples_digit[i]=hamm;
		
	}//end of hamming for
	
	for(i=0;i<=12;i++){
		
		r[i]=calculate_rval(first,last,i);
		if(r[i]<=0){
			return true;
		}
	}//end of r calculation for
	
	//calculation of ai values
	for (j=0;j<=12;j++){
            a[0]=0;
            am1[0]=0;
	}
	
    a[0]=1;
    am1[0]=1;
    km=0;
    em1=r[0];
		
	for (m=1;m<=12;m++){                  //m=2:N+1
        long double err=0.0;                    //err = 0;
        for (j=1;j<=m-1;j++)            //for k=2:m-1
            err += am1[j]*r[m-j];        // err = err + am1(k)*R(m-k+1);
			
        km = (r[m]-err)/em1;            //km=(R(m)-err)/Em1;

		/*if(m==1 && start+1==1)
			cout<<"the km value is "<<km<<endl;*/

        //k[m-1] = long double(km);

		a[m]=(long double)km;                        //am(m)=km;

		/*if(m==1 && start+1==1)
			cout<<"the a[1] value is "<<a[m]<<endl;*/

        for (j=1;j<=m-1;j++)            //for k=2:m-1
             a[j]=long double(am1[j]-km*am1[m-j]);  // am(k)=am1(k)-km*am1(m-k+1);
			 
        em=(1-km*km)*em1;                //Em=(1-km*km)*Em1;
		
        for(s=0;s<=12;s++)                //for s=1:N+1
            am1[s] = a[s];                // am1(s) = am(s)
			
		em1 = em;                        //Em1 = Em;
    }//end of ai calculations
	
	
	//beginning of cepstral co-efficients calculation fout_ci_val
	
	ofstream fout_ci_val;
	fout_ci_val.open(cval_file_str.c_str(),ios::out|ios::app);
	fout_ci_val<<"Frame "<<frame_no<<endl;
	
	c[0]=log(r[0]);
	fout_ci_val<<c[0]<<endl;
	xis[frame_no][1]=c[0];
	
	c[1]=a[1];
	fout_ci_val<<c[1]<<endl;
	xis[frame_no][2]=c[1];
	
	for(i=2;i<=12;i++){
	
		inval=0.0;
		
		for(j=1;j<i;j++){
			
			inval+=long double((j/i))*c[j]*a[i-j];
		
		}
		
		c[i]=a[i]+inval;
		fout_ci_val<<c[i]<<endl;
		xis[frame_no][i+1]=c[i];
		
	}
	
	fout_ci_val<<endl<<endl;
	fout_ci_val.close();


	return false;
}//end of ci_val_func


long double calculate_rval(int first,int last,int i){

	long double sum=0.0;
	int m;
	
	for(m=first;m<=last-i;m++){
	
		sum+=samples_digit[m]*samples_digit[m+i];
		
	}
	
	return (sum/(long double)320);

}//end of func calculate_rval


void initialise_weights(){

	weights[0]=0.;
	weights[1]=1;
	weights[2]=3;
	weights[3]=5;
	weights[4]=9;
	weights[5]=13;
	weights[6]=18;
	weights[7]=25;
	weights[8]=32;
	weights[9]=40;
	weights[10]=49;
	weights[11]=55;
	weights[12]=62;	

}

void create_obs_sequence(){
	ifstream fin_centroids;
	std::string centroids_string;
	long double temp,ci_sum,min_dis,temp_sum;
	int i,j,k,cluster;
	
	fin_centroids.open(centroids_file_str);
	
	if(fin_centroids.is_open()){
		for(i=1;i<K;i++){
			for(j=1;j<ci;j++){
				getline(fin_centroids,centroids_string,' ');
				temp=atof(centroids_string.c_str());
				yis[i][j]=temp;
				/*if(i==32){
					cout<<yis[i][j]<<endl;
				}*/
			}
		}	
	}//end of if
	
	else{
		cout<<"The centroids file is not open"<<endl;
	}
	
	ofstream xis_cluster;
	xis_cluster.open(frames_vectors.c_str(),ios::out|ios::trunc);
	
	for(i=1;i<new_no_of_frames+1;i++){
		for(j=1;j<K;j++){
			ci_sum=0.0;
			for(k=1;k<ci;k++){
				ci_sum+=weights[k]*((xis[i][k+1]-yis[j][k])*(xis[i][k+1]-yis[j][k]));
			}

			if(j==1){
				cluster=j;
				min_dis=ci_sum;
			}
			if(ci_sum<min_dis){
				cluster=j;
				min_dis=ci_sum;
			}
		}//cluster to which xis belongs to is finalised
		
		xis_cluster<<"frame "<<i<<"     Cluster "<<cluster<<endl;
		obs_seq[i]=cluster;
		//cout<<obs_seq[i]<<endl;
	}
	
	xis_cluster.close();
}

void read_a_b_pi(){

	ifstream a_file,b_file,pi_file;
	std::string a_string,b_string,pi_string;
	int i,j;
	//cout<<a_file_str<<endl;
	a_file.open(a_file_str.c_str());
	
	if(!(a_file.is_open())){
		cout<<" a file is not open"<<endl;
		exit(0);
	}
		
		for(i=1;i<N;i++){
			for(j=1;j<N;j++){
				getline(a_file,a_string);
				a[i][j]=atof(a_string.c_str());
				//cout<<a[i][j]<<endl;
				
			}
		}		
		
	//cout<<a[i-1][j-1]<<"last a mat val"<<endl;
	a_file.close();
	
	//reading from b.txt
	
	b_file.open(b_file_str.c_str());
	
	if(!(b_file.is_open())){
		cout<<" b file is not open"<<endl;
		exit(0);
	}
		
		for(i=1;i<N;i++){
			for(j=1;j<K;j++){
				getline(b_file,b_string);
				b[i][j]=atof(b_string.c_str());
				//cout<<b[i][j]<<endl;
				//count++;
				
			}
		}		
	//cout<<b[i-1][j-1]<<"last b mat val"<<endl;
	b_file.close();

	//cout<<"the count is "<<count<<endl;
	
	//reading from pi file
	
	pi_file.open(pi_file_str.c_str());
	
	if(!(pi_file.is_open())){
		cout<<" pi file is not open"<<endl;
		exit(0);
	}
		
		for(i=1;i<N;i++){
			
				getline(pi_file,pi_string);
				pi_mat[i]=atof(pi_string.c_str());
				//cout<<pi_mat[i]<<endl;
				//count++;

		}		
	
	pi_file.close();

}

void train_HMM(){
 
	long double score=0.0;
	int iteration=0;
	
	ofstream a_star_file_out;
	a_star_file_out.open(a_star_file_str.c_str(),ios::out|ios::trunc);
	a_star_file_out.close();
	
	ofstream b_star_file_out;
	b_star_file_out.open(b_star_file_str.c_str(),ios::out|ios::trunc);
	b_star_file_out.close();
	
	ofstream pi_star_file_out;
	pi_star_file_out.open(pi_star_file_str.c_str(),ios::out|ios::trunc);
	pi_star_file_out.close();
	
	ofstream p_star_file_out;
	p_star_file_out.open(p_star_file_str.c_str(),ios::out|ios::trunc);
	p_star_file_out.close();
	
	ofstream q_star_file_out;
	q_star_file_out.open(q_star_file_str.c_str(),ios::out|ios::trunc);
	q_star_file_out.close();

	p_star_old=0.0;
	p_star=1.0;
	
//	while(p_star_old<p_star || iteration==1){
	//while(p_star_old!=p_star ){
	
		iteration++;
		
		score=forward_procedure();
		cout<<score<<endl;
		
		if(score>max_score){
			//cout<<score<<" score"<<endl;
			//cout<<max_score<<" max score"<<endl;
			max_score_digit=curr_digit;
			max_score=score;
		}
		
		/*backward_procedure();
		
		solution_two(); //for gamma calculation
		
		p_star_file_out.open(p_star_file_str.c_str(),ios::out|ios::app);
		p_star_file_out<<"Iteration "<<iteration<<endl;
		
		q_star_file_out.open(q_star_file_str.c_str(),ios::out|ios::app);
		q_star_file_out<<"Iteration "<<iteration<<endl;
		
		p_star_old=p_star;
		p_star=viterbi_algo();
		
		p_star_file_out<<"The score is "<<score<<endl;
		p_star_file_out<<"The P* value is "<<p_star<<endl<<endl;
		//p_star_file_out<<p_star<<endl;
		p_star_file_out.close();
		
		q_star_file_out<<endl<<endl;
		q_star_file_out.close();
		
		a_star_file_out.open(a_star_file_str.c_str(),ios::out|ios::app);
		a_star_file_out<<"Iteration "<<iteration<<endl<<endl;
		
		b_star_file_out.open(b_star_file_str.c_str(),ios::out|ios::app);
		b_star_file_out<<"Iteration "<<iteration<<endl<<endl;
		
		pi_star_file_out.open(pi_star_file_str.c_str(),ios::out|ios::app);
		pi_star_file_out<<"Iteration "<<iteration<<endl<<endl;
		
		solution_three();
		
		a_star_file_out<<endl<<endl;
		a_star_file_out.close();
		
		b_star_file_out<<endl<<endl;
		b_star_file_out.close();
		
		pi_star_file_out<<endl<<endl;
		pi_star_file_out.close();*/

	//}//end of while loop

}//end of train_HMM

long double forward_procedure(){

	//ofstream alpha_file_out;
	int i,j,k,l;
	long double alpha_in_sum=0.0,forward_score=0.0;
	
	//alpha_file_out.open(alpha_file_str.c_str(),ios::out|ios::app);
	
	
	//initialisation
	for(i=1;i<N;i++){
	
		alpha[1][i]=pi_mat[i]*b[i][obs_seq[i]];
	
	}
	
	//induction
	for(i=1;i<new_no_of_frames+1;i++){
		for(j=1;j<N;j++){
		
			alpha_in_sum=0.0;
			
			for(k=1;k<N;k++){
		
				alpha_in_sum+=alpha[i][k]*a[k][j];
	
			}//end of k (N) loop
			
			alpha[i+1][j]=alpha_in_sum*b[j][obs_seq[i+1]];
			
		}//end of j (N) loop
	}//end of T loop
	
	for(i=1;i<N;i++){
	
		forward_score+=alpha[new_no_of_frames][i];
	
	}
	
	/*for(i=1;i<new_no_of_frames+1;i++){
		for(j=1;j<N;j++){
		
			alpha_file_out<<alpha[i][j]<<std::scientific<<"   ";
		
		}
		alpha_file_out<<endl;
	}*/
	
	//cout<<"The score is "<<forward_score<<endl;
	//alpha_file_out.close();

return forward_score;
}//end of forward_procedure



