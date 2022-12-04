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

vector<int> samples_digit;
int no_of_samples;
int no_of_frames;
int new_no_of_frames;
long double yis[K][ci],variance_arr[K][ci],xis[F][ci+1],final_centroids[K][ci];
long double weights[ci];
int xis_clusters[F];

std::string input_file="input\\Universe_S.txt";
std::string norm_file="logs\\Universe\\norm.txt";
std::string cval_file_str="logs\\Universe\\cval.txt";
std::string frame_skip_str="logs\\Universe\\frame_skip.txt";
std::string frames_vectors="logs\\Universe\\xis_clusters.txt";
std::string distortion_file="logs\\Universe\\distortion_iteration.txt";
std::string codebook="logs\\Universe\\codebook.txt";
std::string cluster_count_file="logs\\Universe\\cluster count.txt";
std::string centroids_file_str="logs\\Universe\\Centroids_new.txt";

void normalise();
bool ci_val_func(int,int);
long double calculate_rval(int ,int ,int );
void initialise_weights();
long double build_codebook(int );

int main(){
	bool skip_frame;
	int i,j,count=0,k;
	std::string ci_string;
	
	long double prev_distortion,curr_distortion,initial_sum[ci],split_param[ci],temp_val;
	int m=0,no_of_clusters=1,prev_no_of_clusters;
	
	normalise();
	
	ofstream fout_ci_val;
	fout_ci_val.open(cval_file_str.c_str(),ios::out|ios::trunc);
	fout_ci_val.close();
	
	ofstream fout_frame_skip;
	fout_frame_skip.open(frame_skip_str.c_str(),ios::out|ios::trunc);
	//fout_frame_skip.close();
	
	ofstream codebook_out;
	codebook_out.open(codebook.c_str(),ios::out|ios::trunc);
	codebook_out.close();
	
	ofstream cluster_count_file_out;
	cluster_count_file_out.open(cluster_count_file.c_str(),ios::out|ios::trunc);
	cluster_count_file_out.close();
	
	no_of_frames=(((int)(no_of_samples/320))*4)-4;
	cout<<"The number of frames is "<<no_of_frames<<endl;

	cout<<"Processing"<<endl;
		
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

	//cout<<"The no of frames after skipping is "<<j-1<<endl;
	new_no_of_frames=j-1;
	
	//Starting of the LBG Algorithm implementation
	
	//step 1: computing the centroid of the whole training vectors
	
	for(i=1;i<ci;i++){
	initial_sum[i]=0.0;
	}
	
	for(i=1;i<ci+1;i++){
		for (j=1;j<new_no_of_frames+1;j++){
		
			initial_sum[i]+=xis[j][i+1];
		
		}	
	}
	
	for(i=1;i<ci;i++){
	
		yis[1][i]=initial_sum[i]/long double(new_no_of_frames); //centroid is computed for the whole set of training vectors F (no of frames)
	
	}
	
	//calculating epsilon value (splitting parameter)
	
	//initialising
	for(i=1;i<ci;i++){
	
		variance_arr[1][i]=0.0;
	
	}
	
	for(i=1;i<ci+1;i++){
		for(j=1;j<new_no_of_frames+1;j++){
	
			variance_arr[1][i]+=(xis[j][i+1]-yis[1][i])*(xis[j][i+1]-yis[1][i]);
	
		}
	}
	
	
	for(i=1;i<ci;i++){
	
		variance_arr[1][i]=sqrt (variance_arr[1][i]/long double(new_no_of_frames));
		split_param[i]=variance_arr[1][i]/long double(10);
	
	}
	
	initialise_weights();
	
	ofstream distortion_file_out;
	distortion_file_out.open(distortion_file.c_str(),ios::out|ios::trunc);
	distortion_file_out.close();
	
	codebook_out.open(codebook.c_str(),ios::out|ios::app);
	
		//now once the splitting parameter is decided , splitting the codebook vectors
	while(no_of_clusters<(K-1)){
		
		prev_no_of_clusters=no_of_clusters;
		no_of_clusters=no_of_clusters*2;
		
		cout<<"Number of Clusters "<<no_of_clusters<<endl;
		
		//for(i=1,j=1;i<=prev_no_of_clusters && j<= no_of_clusters; i++,j++){
		for(i=prev_no_of_clusters;i>=1;i--){
		
			for(k=1;k<ci;k++){
				//split_param[k]=(sqrt(variance_arr[i][k]/long double(new_no_of_frames)))/long double(100000);
				split_param[k]=variance_arr[i][k];
				temp_val=yis[i][k];
				yis[i*2][k]=temp_val+split_param[k];
				//j++;
				yis[i*2+1][k]=temp_val-split_param[k];					
			
			}
			//j++;
		
		}//done with the splitting
		
		//classification of the vectors using k means
		
		ofstream distortion_file_out;
		distortion_file_out.open(distortion_file.c_str(),ios::out|ios::app);
		distortion_file_out<<"No of Clusters  "<<"Distortion "<<endl<<endl;
		
	
		cluster_count_file_out.open(cluster_count_file.c_str(),ios::out|ios::app);
		cluster_count_file_out<<"No of Clusters "<<no_of_clusters<<endl;
		
		m=0;
		
		for(i=1,j=1;i<100;i++,j++){
			
			curr_distortion=0.0;
			curr_distortion=build_codebook(no_of_clusters);
			
			if(m==0){
				prev_distortion=curr_distortion;
			}
			else if(prev_distortion==curr_distortion){
				
				i=100;
				ofstream outfile(centroids_file_str);
				for(int x=1; x<K; x++)
				{
				for(int y=1; y<ci; y++)
				{
					outfile<<yis[x][y]<<" ";
				}
				outfile<<"\n";
				}
				outfile.close();
						
			}
			else{
				prev_distortion=curr_distortion;i--;;
			}
			distortion_file_out<<"Iteration "<<j<<endl<<endl;
			distortion_file_out<<no_of_clusters<<"           "<<curr_distortion<<endl;
			m++;
		}
		
		
		distortion_file_out<<endl<<endl<<endl;
		
		
		
		/*for(i=1;i<=no_of_clusters;i++){
		codebook_out<<"Cluster "<<i<<"  "<<endl;
		for(j=1;j<new_no_of_frames+1;j++){
			if(xis_clusters[j]==i){
				codebook_out<<j<<"  ";
			}
		}
			codebook_out<<endl<<endl<<endl;
		}*/
	for(i=1;i<=no_of_clusters;i++){
		codebook_out<<"Cluster "<<i<<"  "<<endl;
		for(j=1;j<new_no_of_frames+1;j++){
			if(xis_clusters[j]==i){
				codebook_out<<j<<"  ";
			}
			
		}
			codebook_out<<endl<<endl<<endl;
			
		}
	//cout<<"This is the end of while loop"<<endl;
	}//end of while loop
	return 0;
}//end of main

//function to normalise
void normalise(){

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
		cout<<"the number of samples is "<<no_of_samples<<endl;
		
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


long double build_codebook(int no_of_clusters){

	//cout<<"entered build codebook proc"<<endl;
	
	int i,j,k,l;
	long double min_dis=0.0,distortion;
	long double ci_sum=0.0,dist_sum=0.0;
	int cluster;
	int cluster_count[K];
	long double cl[K][ci]; //sum of the cis
	long double cl_sq[K][ci]; //sum of the squares of the cis
	
	for(i=1;i<=no_of_clusters;i++){
	
		cluster_count[i]=0;
		
		for(j=1;j<ci;j++){
		
			cl[i][j]=0.0;
			cl_sq[i][j]=0.0;
			variance_arr[i][j]=0.0;
		
		}
	
	}
	
	ofstream xis_cluster;
		xis_cluster.open(frames_vectors.c_str(),ios::out|ios::trunc);
	
	//Calculation of the distance values and finding the min distance
	for(i=1;i<new_no_of_frames+1;i++){ //frames going from 1 to 1000
	
		for(j=1;j<=no_of_clusters;j++){ //clusters going from 1 to current no of clusters
			ci_sum=0.0;
			for(k=1;k<ci;k++){ //ci values going from 1 to 12
		
				ci_sum+=weights[k]*((xis[i][k+1]-yis[j][k])*(xis[i][k+1]-yis[j][k]));
		
			}
			
			//ci_sum=ci_sum/(long double)ci;
			
			if(j==1){
				cluster=j;
				min_dis=ci_sum;
				
				//cout<<"entered min_dis "<<min_dis<<endl;
			}
			if(ci_sum<min_dis){
				
				cluster=j;
				min_dis=ci_sum;
			}
		} //cluster to which xis frame belongs to is finalised
		
		
		xis_cluster<<"frame "<<i<<"     Cluster "<<cluster<<endl;
		xis_clusters[i]=cluster;
		
		
		//ci values going from 1 to 12
			//cout<<"entered this dist_sum proc and cluster is "<<cluster<<endl;
				dist_sum+=min_dis;
			//cout<<dist_sum<<" "<<"k is "<<k<<endl;
			
		
		cluster_count[cluster]=cluster_count[cluster]+1;
		
		for(l=1;l<ci;l++){
		
			cl[cluster][l]+=xis[i][l+1]; //sum of the cis of the cluster
			//cl_sq[cluster][l]+=(xis[i][l]*xis[i][l]);
			variance_arr[cluster][l]+=(xis[i][l+1]-yis[cluster][l])*(xis[i][l+1]-yis[cluster][l]);
		
		}
	
	}	//all the 1000 frames are alloted to some cluster
	
	ofstream cluster_count_file_out;
	cluster_count_file_out.open(cluster_count_file.c_str(),ios::out|ios::app);
	
	for(i=1;i<=no_of_clusters;i++){
	
					cluster_count_file_out<<"Cluster "<<i<<"Count "<<cluster_count[i]<<endl;
	
	}
	
	for(i=1;i<=no_of_clusters;i++){


			if(cluster_count[i]==0){
				cout<<"got a zero cell at cluster "<<i<<endl;
				exit(0);
			}
			
			for(j=1;j<ci;j++){
				final_centroids[i][j]=yis[i][j];
				//yis[i][j]=cl[i][j]/cluster_count[i]; //Code Vectors are updated by calculating the centroid again
				yis[i][j]=cl[i][j]/cluster_count[i]; //Code Vectors are updated by calculating the centroid again
				variance_arr[i][j]=sqrt(variance_arr[i][j]/(new_no_of_frames));
				//variance_arr[i][j]=variance_arr[i][j]/(100*(cluster_count[i]));
				variance_arr[i][j]=variance_arr[i][j]/(10);
			
			}
		
		}
		
	distortion=dist_sum/(long double)(new_no_of_frames);
	cluster_count_file_out<<"Distortion is "<<distortion<<endl<<endl<<endl;
	
	return distortion;
	

}//end of build_codebook