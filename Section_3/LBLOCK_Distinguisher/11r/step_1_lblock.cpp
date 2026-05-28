#include <algorithm>  
#include <iterator>  
#include <bitset>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <iostream>
#include <ctime>
#include <chrono>
#include <random>
#include <cmath> 
#include <fstream>
#include <iomanip>  

using namespace std;
using namespace std::chrono;

typedef uint16_t u16;
typedef uint8_t u8;



std::string generateFileName(int train_num, int R, int choose_block) {
    std::time_t now = std::time(0);
    std::stringstream fileName;
    fileName << "cipher_txt_npy/" <<  "cipher_" << choose_block << "_block_" << R << "r_" << "2_" << train_num << "_"<< now << ".txt";  
    return fileName.str();
}



#define KSIZE 80

static const u8 S0[16] = {14, 9, 15, 0, 13, 4, 10, 11, 1, 2, 8, 3, 7, 6, 12, 5};
static const u8 S1[16] = {4, 11, 14, 9, 15, 13, 0, 10, 7, 12, 5, 6, 2, 8, 1, 3};
static const u8 S2[16] = {1, 14, 7, 12, 15, 13, 0, 6, 11, 5, 9, 3, 2, 4, 8, 10};
static const u8 S3[16] = {7, 6, 8, 11, 0, 15, 3, 14, 9, 10, 12, 13, 5, 2, 4, 1};
static const u8 S4[16] = {14, 5, 15, 0, 7, 2, 12, 13, 1, 8, 4, 9, 11, 10, 6, 3};
static const u8 S5[16] = {2, 13, 11, 12, 15, 14, 0, 9, 7, 10, 6, 3, 1, 8, 4, 5};
static const u8 S6[16] = {11, 9, 4, 14, 0, 15, 10, 13, 6, 12, 5, 7, 3, 8, 1, 2};
static const u8 S7[16] = {13, 10, 15, 0, 14, 4, 9, 11, 2, 1, 8, 3, 7, 5, 12, 6};
static const u8 S8[16] = {8, 7, 14, 5, 15, 13, 0, 6, 11, 12, 9, 10, 2, 4, 1, 3};
static const u8 S9[16] = {11, 5, 15, 0, 7, 2, 9, 13, 4, 8, 1, 12, 14, 10, 3, 6};


 void EncryptKeySchedule(u8 key[10], u8 output[][4])
{
     u8 i, KeyR[4];
     
     output[0][3] = key[9];
     output[0][2] = key[8];
     output[0][1] = key[7];
     output[0][0] = key[6];
     
     for(i=1;i<32;i++)
     {
                   
     KeyR[3]=key[9];
     KeyR[2]=key[8];
     KeyR[1]=key[7];     
     KeyR[0]=key[6];     
     
     key[9]=(((key[6] & 0x07)<<5)&0xE0) ^ (((key[5]& 0xF8)>>3) & 0x1F);
     key[8]=(((key[5] & 0x07)<<5)&0xE0) ^ (((key[4]& 0xF8)>>3) & 0x1F);
     key[7]=(((key[4] & 0x07)<<5)&0xE0) ^ (((key[3]& 0xF8)>>3) & 0x1F);
     key[6]=(((key[3] & 0x07)<<5)&0xE0) ^ (((key[2]& 0xF8)>>3) & 0x1F);
     key[5]=(((key[2] & 0x07)<<5)&0xE0) ^ (((key[1]& 0xF8)>>3) & 0x1F);
     key[4]=(((key[1] & 0x07)<<5)&0xE0) ^ (((key[0]& 0xF8)>>3) & 0x1F);
     key[3]=(((key[0] & 0x07)<<5)&0xE0) ^ (((KeyR[3]& 0xF8)>>3) & 0x1F);
     key[2]=(((KeyR[3] & 0x07)<<5)&0xE0) ^ (((KeyR[2]& 0xF8)>>3) & 0x1F);
     key[1]=(((KeyR[2] & 0x07)<<5)&0xE0) ^ (((KeyR[1]& 0xF8)>>3) & 0x1F);
     key[0]=(((KeyR[1] & 0x07)<<5)&0xE0) ^ (((KeyR[0]& 0xF8)>>3) & 0x1F);         
                    
                     
     key[9]=(S9[((key[9]>>4) & 0x0F)]<<4) ^ S8[(key[9]& 0x0F)];
     
     key[6]=key[6] ^ ((i>>2) & 0x07);
     key[5]=key[5] ^ ((i & 0x03)<<6);
        
     output[i][3] = key[9];
     output[i][2] = key[8];
     output[i][1] = key[7];
     output[i][0] = key[6];                      
     }                          
}


void Swap(u8 block[8])
{
    u8 tmp[4];

    tmp[0] = block[0];
    tmp[1] = block[1];
    tmp[2] = block[2];
    tmp[3] = block[3];

    block[0] = block[4];
    block[1] = block[5];
    block[2] = block[6];
    block[3] = block[7];

    block[4] = tmp[0];
    block[5] = tmp[1];
    block[6] = tmp[2];
    block[7] = tmp[3];
}

 void OneRound(u8 x[8], u8 k[4])
{
	u8 t[4],tmp[4];

	
    tmp[0]=x[4]^k[0];
    tmp[1]=x[5]^k[1];
    tmp[2]=x[6]^k[2];
    tmp[3]=x[7]^k[3];         

    
    tmp[0] = ((S1[((tmp[0])>>4) & 0x0F])<<4)^S0[(tmp[0] & 0x0F)];
    tmp[1] = ((S3[((tmp[1])>>4) & 0x0F])<<4)^S2[(tmp[1] & 0x0F)];
    tmp[2] = ((S5[((tmp[2])>>4) & 0x0F])<<4)^S4[(tmp[2] & 0x0F)];
    tmp[3] = ((S7[((tmp[3])>>4) & 0x0F])<<4)^S6[(tmp[3] & 0x0F)];          
    
    
	t[0] =((tmp[0]>>4) & 0x0F)^(tmp[1] & 0xF0);
	t[1] = (tmp[0] & 0x0F) ^ ((tmp[1]& 0x0F)<<4);
	t[2] = ((tmp[2]>>4) & 0x0F)^(tmp[3] & 0xF0);
	t[3] = (tmp[2] & 0x0F) ^ ((tmp[3]& 0x0F)<<4);
    

    
    tmp[0]=x[3]^t[0]; 
    tmp[1]=x[0]^t[1]; 
    tmp[2]=x[1]^t[2]; 
    tmp[3]=x[2]^t[3]; 
    
	
    x[0]=tmp[0];
    x[1]=tmp[1];
    x[2]=tmp[2];
    x[3]=tmp[3];      
    
  
}

 void Encrypt(int R, u8 x[8], u8 subkey[][4])
{
     int i;
     
     for(i=0; i<R-1; i++)
     {
        OneRound(x, subkey[i]);        
        Swap(x);
     }
     
     if(i  == 31)
     {
        OneRound(x, subkey[i]);
     }
     
}

 void OneRound_Inv(u8 y[8], u8 k[4])
{
    u8 t[4],tmp[4];


    tmp[0]=y[4]^k[0]; 
    tmp[1]=y[5]^k[1]; 
    tmp[2]=y[6]^k[2]; 
    tmp[3]=y[7]^k[3];  
     
 

    tmp[0] = ((S1[((tmp[0])>>4) & 0x0F])<<4)^S0[(tmp[0] & 0x0F)];
    tmp[1] = ((S3[((tmp[1])>>4) & 0x0F])<<4)^S2[(tmp[1] & 0x0F)];
    tmp[2] = ((S5[((tmp[2])>>4) & 0x0F])<<4)^S4[(tmp[2] & 0x0F)];
    tmp[3] = ((S7[((tmp[3])>>4) & 0x0F])<<4)^S6[(tmp[3] & 0x0F)];    
 

	t[0] =((tmp[0]>>4) & 0x0F)^(tmp[1] & 0xF0);
	t[1] = (tmp[0] & 0x0F) ^ ((tmp[1]& 0x0F)<<4);
	t[2] = ((tmp[2]>>4) & 0x0F)^(tmp[3] & 0xF0);
	t[3] = (tmp[2] & 0x0F) ^ ((tmp[3]& 0x0F)<<4);

	tmp[0]= y[0]^t[0]; 
    tmp[1]= y[1]^t[1]; 
    tmp[2]= y[2]^t[2]; 
    tmp[3]= y[3]^t[3]; 
 

    y[0]=tmp[1];
    y[1]=tmp[2];
    y[2]=tmp[3];
    y[3]=tmp[0];
 
}

 void Decrypt(u8 x[8], u8 subkey[][4])
{
     int i;

     OneRound_Inv(x, subkey[31]);
     for(i=30; i>=0; i--)
     {
        Swap(x);
        OneRound_Inv(x, subkey[i]);   
     }
}


void enc_package(int R,  uint8_t ciphertext_xor[16][16], int choose_block, uint8_t  num_and_key[2][16])
{

    
    uint8_t ciphertext_1[16];
    uint8_t ciphertext_2[16];

    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 g2(seed1); 

    uint64_t plaintext_1_random = g2();
    uint64_t key_1_random = g2();
    uint64_t key_2_random = static_cast<uint16_t>(g2());
    
    u8 rkey[32][4];
    u8 state[8];
    u8 mkey[10]; 
    uint8_t plaintext_1[16];
    for (uint8_t i = 0; i < 16; i++) {
        plaintext_1[i] = static_cast<uint8_t>(plaintext_1_random >> (4 * (15 - i)) & 0xF);  
    }

    for (uint8_t i = 0; i < 8; i++) {
        mkey[i] = static_cast<uint8_t>(key_1_random >> (8 * (7 - i)) & 0xFF); 
    }

    mkey[8] = static_cast<uint8_t>(key_2_random >> (8 * (1 )) & 0xFF);
    mkey[9] = static_cast<uint8_t>(key_2_random >> (8 * (0 )) & 0xFF);
    EncryptKeySchedule(mkey,rkey);


    EncryptKeySchedule(mkey,rkey);

    for( uint8_t i =0; i < 16; i ++)
    {
        plaintext_1[choose_block] = i;

        for(int k = 7; k >= 0; k --)
        {
            state[k] = ((plaintext_1[15- 2* k - 1]<<4)&0xF0) ^ ((plaintext_1[15- 2* k ] & 0x0F));
        }

        Encrypt(R, state, rkey);

        for(int j = 7; j >= 0; j --)
        {
            ciphertext_xor[i][15- 2* j - 1] = static_cast<int>(state[j] >> 4);
            ciphertext_xor[i][15- 2* j   ] =  static_cast<int>(state[j] & 0x0f);

        }



    }

}

int main()
{

    int R = 11;
    int choose_block = 8;
    int train_num = 19;
    uint8_t ciphertext_no_xor[16][16]; 
    uint8_t num_and_key[2][16];

    double repeat_num = pow(2, train_num);
    std::string fileName = generateFileName(train_num,  R, choose_block);


    ofstream outfile(fileName,  ios::app);

    auto start = high_resolution_clock::now();
    int block[]= {1, 9, 11};

    for (int i = 0 ; i < repeat_num; i ++)
    {

        enc_package(R,   ciphertext_no_xor, choose_block,  num_and_key);

        int xor_num = 0;

        for( int k = 0; k < 16; k ++)  
        {
            
            for(int m = k  + 1; m < 16; m ++)   
            {


                for(int c = 0; c < 16; c++) 
                {
                    auto it = std::find(std::begin(block), std::end(block), c);

                    if (it != std::end(block)) {
                    int temp = ciphertext_no_xor[k][c]  ^ ciphertext_no_xor[m][c] ;

                    outfile <<  static_cast<int>(std::bitset<4>(temp)[3] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[2] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[1] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[0] ) << " ";
                    }
                }

            }

    }
    }
    outfile << endl;

    outfile.close();

    auto end = high_resolution_clock::now();

    auto duration = duration_cast<std::chrono::seconds>(end - start);  


	 


}