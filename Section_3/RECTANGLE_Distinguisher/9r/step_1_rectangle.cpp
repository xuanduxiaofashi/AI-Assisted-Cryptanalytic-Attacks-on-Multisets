#include <string.h>
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
#define BLOCK_SIZE 8
#define KEY_SIZE 10
#define ROUND_KEYS_SIZE 208
#define NUMBER_OF_ROUNDS 25

#define ROTL1(x)    (((x) << 1) | ((x) >> 15))
#define ROTL12(x)   (((x) << 12) | ((x) >> 4))
#define ROTL13(x)   (((x) << 13) | ((x) >> 3))

#define ROTR1(x)	(((x)<<15)^((x)>>1))
#define ROTR12(x)	(((x)<<4)^((x)>>12))
#define ROTR13(x)	(((x)<<3)^((x)>>13))

#define READ_ROUND_CONSTANT_BYTE(x) (x)
#define READ_ROUND_KEY_BYTE(x) (x)
#define READ_ROUND_KEY_WORD(x) (x)



std::string generateFileName(int train_num, int R, int choose_block) {
    std::time_t now = std::time(0);
    std::stringstream fileName;
    fileName << "cipher_txt_npy/" <<  "cipher_" << choose_block << "_block_" << R << "r_" << "2_" << train_num << "_"<< now << ".txt";  
    return fileName.str();
}
void IS_layer(uint16_t *data){
   uint16_t  temp[5];
   temp[0] = data[0]|data[3]; 
   temp[1] = data[0]^data[3]; 
   temp[2] = data[0]&(data[2]^0xFFFF); 
   temp[3] = data[2]^temp[0]; 
   data[2] = data[1]^temp[3]; 
   temp[4] = temp[3]^0xffff; 
   temp[0] = data[3]^data[2]; 
   data[1] = temp[2]^temp[0]; 
   temp[2] = temp[4]|temp[0]; 
   data[3] = temp[1]^temp[2]; 
   temp[0] = data[1]&(data[3]^0xffff);   
   data[0] = temp[4]^temp[0];
}

void IP_layer(uint16_t *data){
   data[1] = ROTR1(data[1]);
   data[2] = ROTR12(data[2]);
   data[3] = ROTR13(data[3]);
}

void Iround_function(uint16_t *data,uint16_t *rkey){
   uint8_t i;
   for(i=0;i<4;i++) data[i] ^= READ_ROUND_KEY_WORD(rkey[i]);
   IP_layer(data);
   IS_layer(data);
}


void Decrypt(uint8_t *block, uint8_t *roundKeys)
{
        uint8_t i;
        for(i=NUMBER_OF_ROUNDS;i!=0;i--) Iround_function((uint16_t*)block,(uint16_t*)(roundKeys+8*i));
        for(i=0;i<8;i++) block[i] ^= READ_ROUND_KEY_BYTE(roundKeys[i]);
}



uint8_t round_constants[25] = {
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16, 0x0C, 0x19,
    0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C, 0x18, 0x11, 0x03, 0x06,
    0x0D, 0x1B, 0x17, 0x0E, 0x1D
};


void S_layer(uint16_t *data) {
    uint16_t temp[5];
    temp[0] = data[1] ^ 0xffff;
    temp[1] = data[0] & temp[0];
    temp[2] = data[2] ^ data[3];
    temp[3] = temp[1] ^ temp[2];
    temp[1] = data[3] | temp[0];
    temp[0] = data[0] ^ temp[1];
    temp[1] = data[2] ^ temp[0];
    temp[4] = data[1] ^ data[2];
    data[3] = temp[2] & temp[0];
    data[3] ^= temp[4];
    data[2] = temp[3] | temp[4];
    data[2] ^= temp[0];
    data[0] = temp[3];
    data[1] = temp[1];
}


void RunEncryptionKeySchedule(uint8_t *key, uint8_t *roundKeys) {
    uint8_t i, j, key_state[10] = {0}, temp[8];
    uint16_t temp16;
    memcpy(key_state, key, 10);

    for (i = 0; i < NUMBER_OF_ROUNDS; i++) {
        for (j = 0; j < 8; j++) roundKeys[i * 8 + j] = key_state[j];

        for (j = 0; j < 4; j++) temp[2 * j + 1] = key_state[2 * j + 1];
        S_layer((uint16_t *)temp);

        for (j = 0; j < 4; j++) {
            key_state[2 * j + 1] &= 0xF0;
            key_state[2 * j + 1] ^= temp[2 * j + 1] & 0x0F;
        }

        temp[0] = key_state[0];
        temp[1] = key_state[1];
        key_state[0] = key_state[1] ^ key_state[2];
        key_state[1] = temp[0] ^ key_state[3];
        key_state[2] = key_state[4];
        key_state[3] = key_state[5];
        key_state[4] = key_state[6];
        key_state[5] = key_state[7];

        temp16 = ((uint16_t)key_state[6] << 8) | key_state[7];
        temp16 = ROTL12(temp16);

        key_state[6] = (temp16 >> 8) ^ key_state[8];
        key_state[7] = (temp16 & 0xFF) ^ key_state[9];
        key_state[8] = temp[0];
        key_state[9] = temp[1];

        key_state[1] ^= READ_ROUND_CONSTANT_BYTE(round_constants[i]);
    }

    for (j = 0; j < 8; j++) roundKeys[NUMBER_OF_ROUNDS * 8 + j] = key_state[j];
}


void P_layer(uint16_t *data) {
    data[1] = ROTL1(data[1]);
    data[2] = ROTL12(data[2]);
    data[3] = ROTL13(data[3]);
}


void round_function(uint16_t *data, uint16_t *rkey) {
    for (int i = 0; i < 4; i++) data[i] ^= READ_ROUND_KEY_WORD(rkey[i]);
    S_layer(data);
    P_layer(data);
}


void Encrypt(uint8_t *block, uint8_t *roundKeys, int R) {
    for (int i = 0; i < R; i++) {
        round_function((uint16_t *)(block), (uint16_t *)(roundKeys + 8 * i));
    }
    if(R == NUMBER_OF_ROUNDS)
    {
        for (int i = 0; i < 8; i++) block[i] ^= READ_ROUND_KEY_BYTE(roundKeys[8 * NUMBER_OF_ROUNDS + i]);
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
    uint8_t roundKeys[ROUND_KEYS_SIZE];
    uint8_t block[BLOCK_SIZE];
    uint8_t key[KEY_SIZE];




    uint8_t plaintext_1[16];
    for (uint8_t i = 0; i < 16; i++) {
        plaintext_1[i] = static_cast<uint8_t>(plaintext_1_random >> (4 * (15 - i)) & 0xF);  
    }

    for (uint8_t i = 0; i < 8; i++) {
        key[i] = static_cast<uint8_t>(key_1_random >> (8 * (7 - i)) & 0xFF);  
    }

    key[8] = static_cast<uint8_t>(key_2_random >> (8 * (1 )) & 0xFF);
    key[9] = static_cast<uint8_t>(key_2_random >> (8 * (0 )) & 0xFF);



    RunEncryptionKeySchedule(key, roundKeys);


    for( uint8_t i =0; i < 16; i ++)
    {

        std::bitset<4> binary_list(i);
        int select_num = int(choose_block / 4);

        int choose_bit = 3 - int(choose_block % 4);

        plaintext_1[select_num] =  plaintext_1[select_num] & (~(1 <<  choose_bit)) | (binary_list[3] << choose_bit);
        plaintext_1[select_num + 4] = plaintext_1[select_num + 4] & (~(1 <<   choose_bit)) | (binary_list[2] << choose_bit);;
        plaintext_1[select_num + 8] = plaintext_1[select_num + 8] & (~(1 <<   choose_bit)) | (binary_list[1] << choose_bit);;
        plaintext_1[select_num + 12] = plaintext_1[select_num + 12] & (~(1 <<   choose_bit)) | (binary_list[0] << choose_bit);;
        for(int k = 0; k <= 7; k ++)
        {
            block[k] = ((plaintext_1[2* k ]<<4)&0xF0) ^ ((plaintext_1[2* k + 1] & 0x0F));
        }

        Encrypt(block, roundKeys, R);
        IP_layer((uint16_t*)block);
        IS_layer((uint16_t*)block);

        for(int j = 0; j < 8; j ++)
        {
            ciphertext_xor[i][0 + 2* j  ] = static_cast<int>(block[j] >> 4);
            ciphertext_xor[i][1 + 2* j   ] =  static_cast<int>(block[j] & 0x0f);
        }

    }

}





int main()
{

    int R = 9;
    int choose_block = 13;
    int train_num = 17;
    uint8_t ciphertext_no_xor[16][16]; 
    uint8_t num_and_key[2][16];

    double repeat_num = pow(2, train_num);

    std::string fileName = generateFileName(train_num,  R, choose_block);


    ofstream outfile(fileName,  ios::app);

    auto start = high_resolution_clock::now();


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
                    int temp = ciphertext_no_xor[k][c]  ^ ciphertext_no_xor[m][c] ;

                    outfile <<  static_cast<int>(std::bitset<4>(temp)[3] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[2] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[1] ) << " ";
                    outfile <<  static_cast<int>(std::bitset<4>(temp)[0] ) << " ";

                }

            }

        }
        outfile << endl;
    }
    


    outfile.close();

    auto end = high_resolution_clock::now();


	 


}

