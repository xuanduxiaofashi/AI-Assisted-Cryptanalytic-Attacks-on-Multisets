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

const uint8_t S[16] = {0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf};

const uint8_t Sinv[16] = {0x3, 0x4, 0x6, 0x8, 0xc, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf};

const uint8_t P[16] = {0x0, 0x1, 0x2, 0x3, 0x7, 0x4, 0x5, 0x6, 0xa, 0xb, 0x8, 0x9, 0xd, 0xe, 0xf, 0xc};
const uint8_t Pinv[16] = {0x0, 0x1, 0x2, 0x3, 0x5, 0x6, 0x7, 0x4, 0xa, 0xb, 0x8, 0x9, 0xf, 0xc, 0xd, 0xe};

const uint8_t Q[16] = {0x9, 0xf, 0x8, 0xd, 0xa, 0xe, 0xc, 0xb, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7};

const uint8_t RC[36] = {0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F,
                        0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B,
                        0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E,
                        0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D};

void print_state(uint8_t state[16]);
void convert_hexstr_to_statearray(string hex_str, uint8_t int_array[16], bool reversed);
uint8_t tweak_tk2_lfsr(uint8_t x);
void mix_columns(uint8_t state[16]);
void inv_mix_columns(uint8_t state[16]);
void tweakey_schedule(int rounds, uint8_t tk1[][16], uint8_t tk2[][16], uint8_t round_tweakey[][8]);
void enc(int R, uint8_t plaintext[16], uint8_t ciphertext[16], uint8_t tk[][8], uint8_t num_and_key[][16]);
void dec(int R, uint8_t plaintext[16], uint8_t ciphertext[16], uint8_t tk[][8]);

std::string generateFileName(int train_num, int R, int choose_block) {
    std::time_t now = std::time(0);
    std::stringstream fileName;
    fileName << "cipher_txt_npy/" <<  "cipher_" << choose_block << "_block_" << R << "r_" << "2_" << train_num << "_"<< now << ".txt";  
    return fileName.str();
}


void print_state(uint8_t state[16])
{
    for (int i = 0; i < 16; i++)
        printf("%01x", state[i]);
    printf("\n");
}

void convert_hexstr_to_statearray(string hex_str, uint8_t int_array[16], bool reversed = false)
{
    if (reversed == true)
        for (size_t i = 0; i < 16; i++)
            int_array[15 - i] = static_cast<uint8_t>(stoi(hex_str.substr(i, 1), 0, 16) & 0xf);
    else
        for (size_t i = 0; i < 16; i++)
            int_array[i] = static_cast<uint8_t>(stoi(hex_str.substr(i, 1), 0, 16) & 0xf);
}

uint8_t tweak_tk2_lfsr(uint8_t x)
{
    x = (x << 1) ^ ((x >> 3) & 0x1) ^ ((x >> 2) & 0x1);
    x = x & 0xf;
    return x;
}

void mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++)
    {
        state[j + 4 * 1] ^= state[j + 4 * 2];
        state[j + 4 * 2] ^= state[j + 4 * 0];
        state[j + 4 * 3] ^= state[j + 4 * 2];
        tmp = state[j + 4 * 3];
        state[j + 4 * 3] = state[j + 4 * 2];
        state[j + 4 * 2] = state[j + 4 * 1];
        state[j + 4 * 1] = state[j + 4 * 0];
        state[j + 4 * 0] = tmp;
    }
}

void inv_mix_columns(uint8_t state[16])
{
    uint8_t tmp;
    for (uint8_t j = 0; j < 4; j++)
    {
        tmp = state[j + 4 * 3];
        state[j + 4 * 3] = state[j + 4 * 0];
        state[j + 4 * 0] = state[j + 4 * 1];
        state[j + 4 * 1] = state[j + 4 * 2];
        state[j + 4 * 2] = tmp;
        state[j + 4 * 3] ^= state[j + 4 * 2];
        state[j + 4 * 2] ^= state[j + 4 * 0];
        state[j + 4 * 1] ^= state[j + 4 * 2];
    }
}

void tweakey_schedule(int rounds, uint8_t tk1[][16], uint8_t tk2[][16], uint8_t round_tweakey[][8])
{
   
    uint8_t tkp1[rounds - 1][16];
    uint8_t tkp2[rounds - 1][16];
    for (uint8_t i = 0; i < 16; i++)
        tk1[0][i] = (tk1[0][i] & 0xf);
    for (uint8_t i = 0; i < 16; i++)
        tk2[0][i] = (tk2[0][i] & 0xf);
    for (uint8_t i = 0; i < 8; i++)
        round_tweakey[0][i] = (tk1[0][i] ^ tk2[0][i]);
    for (int r = 1; r < rounds; r++)
    {
    
        for (int i = 0; i < 16; i++)
        {
            tkp1[r - 1][i] = tk1[r - 1][Q[i]];
            tkp2[r - 1][i] = tk2[r - 1][Q[i]];
        }
      
        for (int i = 0; i < 16; i++)
        {
           
            tk1[r][i] = tkp1[r - 1][i];
            if (i < 8)
            {
                tk2[r][i] = tweak_tk2_lfsr(tkp2[r - 1][i]);
            }
            else
            {
                tk2[r][i] = tkp2[r - 1][i];
            }
        }
  
        for (int i = 0; i < 8; i++)
            round_tweakey[r][i] = (tk1[r][i] ^ tk2[r][i]);
 
    }
}

void enc_package(int R,  uint8_t ciphertext_xor[16][16], int choose_block, uint8_t  num_and_key[2][16])
{

    uint8_t tk1[R][16];
    uint8_t tk2[R][16];
    uint8_t rtk[R][8];
    uint8_t tweakey_1[16];
    uint8_t tweakey_2[16];
    uint8_t plaintext_1[16];


    uint8_t ciphertext_1[16];
    uint8_t ciphertext_2[16];


    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 g2(seed1); 

    uint64_t plaintext_1_random = g2();
    uint64_t tweakey_1_random = g2();
    uint64_t tweakey_2_random = g2();



    for (uint8_t i = 0; i < 16; i++) {
        plaintext_1[i] = static_cast<uint8_t>(plaintext_1_random >> (4 * (15 - i)) & 0xF);  
        tweakey_1[i] = static_cast<uint8_t>(tweakey_1_random >> (4 * (15 - i)) & 0xF); 
        tweakey_2[i] = static_cast<uint8_t>(tweakey_2_random >> (4 * (15 - i)) & 0xF); \
    }




    for (uint8_t i = 0; i < 16; i++)
    {
        tk1[0][i] = tweakey_1[i];
        tk2[0][i] = tweakey_2[i];
    }
    tweakey_schedule(R + 1, tk1, tk2, rtk);


   


    for( int i = 0; i < 16; i ++)
    {
        plaintext_1[choose_block] = i;


        enc(R, plaintext_1, ciphertext_1, rtk,  num_and_key);

        for(int j = 0; j < 16; j ++)
        {
            ciphertext_xor[i][j] = ciphertext_1[j];

        }
    }




  

}


void enc(int R, uint8_t plaintext[16], uint8_t ciphertext[16], uint8_t tk[][8], uint8_t num_and_key[][16])
{
    for (uint8_t i = 0; i < 16; i++)
    {
        ciphertext[i] = plaintext[i] & 0xf;
    }
    for (uint8_t r = 0; r < R; r++)
    {

        for (uint8_t i = 0; i < 16; i++)
            ciphertext[i] = S[ciphertext[i]];
   
        ciphertext[0] ^= (RC[r] & 0xf);
        ciphertext[4] ^= ((RC[r] >> 4) & 0x3);
        ciphertext[8] ^= 0x2;

        for (uint8_t i = 0; i < 8; i++)
            ciphertext[i] ^= tk[r][i];

        if(r  == R-1)
        {
     
            num_and_key[0][0] = (RC[r] & 0xf);
            num_and_key[0][1] = (RC[r] >> 4) & 0x3;
            num_and_key[0][2] = 0x2;
            for (uint8_t i = 0; i < 8; i++)
                num_and_key[1][i] = tk[r][i];
        }
   
        uint8_t temp[16];
        for (uint8_t i = 0; i < 16; i++)
            temp[i] = ciphertext[i];
        for (uint8_t i = 0; i < 16; i++)
            ciphertext[i] = temp[P[i]];

        mix_columns(ciphertext);

    }
}

void dec(int R, uint8_t plaintext[16], uint8_t ciphertext[16], uint8_t tk[][8])
{
    for (uint8_t i = 0; i < 16; i++)
    {
        plaintext[i] = ciphertext[i] & 0xf;
    }
    int ind;
    uint8_t temp[16];
    for (int r = 0; r < R; r++)
    {

        inv_mix_columns(plaintext);
   
        for (uint8_t i = 0; i < 16; i++)
            temp[i] = plaintext[i];
        for (uint8_t i = 0; i < 16; i++)
            plaintext[i] = temp[Pinv[i]];

        ind = R - r - 1;
        for (uint8_t i = 0; i < 8; i++)
            plaintext[i] ^= tk[ind][i];

        plaintext[0] ^= (RC[ind] & 0xf);
        plaintext[4] ^= ((RC[ind] >> 4) & 0x3);
        plaintext[8] ^= 0x2;

        for (uint8_t i = 0; i < 16; i++)
            plaintext[i] = Sinv[plaintext[i]];

    }
}


int main()
{



    int R = 7;
    int choose_block = 10;
    int train_num = 17;

    int temp_1, temp_2, temp_3;

    uint8_t ciphertext_no_xor[16][16]; 
    uint8_t num_and_key[2][16];

    int temp_1_3, temp_1_2, temp_1_1, temp_1_0;

    int temp_2_3, temp_2_2, temp_2_1, temp_2_0;
    int temp_3_3, temp_3_2, temp_3_1, temp_3_0;
    double repeat_num = pow(2, train_num) * 54 ;


    std::string fileName = generateFileName(train_num,  R, choose_block);

    ofstream outfile(fileName,  ios::app);

    auto start = high_resolution_clock::now();


    
    for (int i = 0 ; i < repeat_num; i ++)
    {

        

        enc_package(R,   ciphertext_no_xor, choose_block,  num_and_key);
        int xor_num = 0;

        for( int k = 0; k < 16; k ++)// mask1
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_3 ^ cp_x_9_0 ^ cp_x_13_0            ) << " ";

                      
        }


        for( int k = 0; k < 16; k ++)// mask2
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_3 ^ cp_x_9_0 ^ cp_x_13_0 ^ cp_x_9_2 ^ cp_x_13_2         ) << " ";

                      
        }



        for( int k = 0; k < 16; k ++)// mask3
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_3 ^ cp_x_9_0 ^ cp_x_13_0 ^ cp_x_9_3 ^ cp_x_13_3   ) << " ";

                      
        }



        for( int k = 0; k < 16; k ++)// mask4
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_3 ^ cp_x_9_0 ^ cp_x_13_0 ^ cp_x_9_2 ^ cp_x_13_2 ^ cp_x_9_3 ^ cp_x_13_3 ) << " ";

                      
        }



        for( int k = 0; k < 16; k ++)// mask5
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_2 ^ cp_x_9_3 ^ cp_x_13_3 ) << " ";

                      
        }

        for( int k = 0; k < 16; k ++)// mask6
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_2 ^ cp_x_9_3 ^ cp_x_9_2 ^ cp_x_13_3 ^ cp_x_13_2 ) << " ";

                      
        }

        for( int k = 0; k < 16; k ++)// mask7
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_2 ^ cp_x_9_3 ^ cp_x_9_1 ^ cp_x_13_3 ^ cp_x_13_1 ) << " ";

                      
        }

        for( int k = 0; k < 16; k ++)// mask8
        {

            temp_1 = ciphertext_no_xor[k][6] ^  ciphertext_no_xor[k][10] ^ ciphertext_no_xor[k][14];
            temp_2 = ciphertext_no_xor[k][0] ^  ciphertext_no_xor[k][12];
            temp_3 = ciphertext_no_xor[k][7] ^  ciphertext_no_xor[k][15];

            temp_1_3 =  static_cast<int>(std::bitset<4>(temp_1)[3] );
            temp_1_2 =  static_cast<int>(std::bitset<4>(temp_1)[2] );
            temp_1_1 =  static_cast<int>(std::bitset<4>(temp_1)[1] );
            temp_1_0 =  static_cast<int>(std::bitset<4>(temp_1)[0] );

            temp_2_3 =  static_cast<int>(std::bitset<4>(temp_2)[3] );
            temp_2_2 =  static_cast<int>(std::bitset<4>(temp_2)[2] );
            temp_2_1 =  static_cast<int>(std::bitset<4>(temp_2)[1] );
            temp_2_0 =  static_cast<int>(std::bitset<4>(temp_2)[0] );

            temp_3_3 =  static_cast<int>(std::bitset<4>(temp_3)[3] );
            temp_3_2 =  static_cast<int>(std::bitset<4>(temp_3)[2] );
            temp_3_1 =  static_cast<int>(std::bitset<4>(temp_3)[1] );
            temp_3_0 =  static_cast<int>(std::bitset<4>(temp_3)[0] );

            int cp_x_9_3 = temp_3_2 ^ temp_3_3 ^ (temp_3_1 & temp_3_0) ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_2 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_1);
            int cp_x_9_2 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_3 & temp_3_0) ^ (temp_3_3 & temp_3_2);
            int cp_x_9_1 = temp_3_0 ^ (temp_3_3 & temp_3_2) ^ temp_3_3 ^ temp_3_2;
            int cp_x_9_0 = temp_3_0 ^ temp_3_1 ^ temp_3_2 ^ (temp_3_1 & temp_3_0) ^ (temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_1) ^ (temp_3_3 & temp_3_2 & temp_3_0) ^ (temp_3_3 & temp_3_2 & temp_3_1);

            int cp_x_13_3 = temp_2_2 ^ temp_2_3 ^ (temp_2_1 & temp_2_0) ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_2 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_1);
            int cp_x_13_2 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_3 & temp_2_0) ^ (temp_2_3 & temp_2_2);
            int cp_x_13_1 = temp_2_0 ^ (temp_2_3 & temp_2_2) ^ temp_2_3 ^ temp_2_2;
            int cp_x_13_0 = temp_2_0 ^ temp_2_1 ^ temp_2_2 ^ (temp_2_1 & temp_2_0) ^ (temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_1) ^ (temp_2_3 & temp_2_2 & temp_2_0) ^ (temp_2_3 & temp_2_2 & temp_2_1) ;
      
            outfile <<  ( temp_1_2 ^ cp_x_9_3 ^ cp_x_9_2 ^ cp_x_9_1 ^ cp_x_13_3 ^ cp_x_13_2 ^ cp_x_13_1 ) << " ";

                      
        }


        outfile << endl;


    }

    outfile.close();

    auto end = high_resolution_clock::now();
    

}