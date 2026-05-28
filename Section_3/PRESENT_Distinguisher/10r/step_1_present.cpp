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

std::string generateFileName(int train_num, int R, int choose_block) {
    std::time_t now = std::time(0);
    std::stringstream fileName;
    fileName << "cipher_txt_npy/" <<  "cipher_" << choose_block << "_block_" << R << "r_" << "2_" << train_num << "_"<< now << ".txt"; 
    return fileName.str();
}


struct byte {
    uint8_t nibble1 : 4;
    uint8_t nibble2 : 4;
} __attribute__((__packed__));

uint8_t S[] = {0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2};
uint8_t invS[] = {0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA};
uint8_t P[] = {
    0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
    4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
    8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
    12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63
};

byte* fromHexStringToBytes(const string &block) {
    byte* bytes = new byte[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i].nibble1 = (block[2*i] >= '0' && block[2*i] <= '9') ? block[2*i] - '0' : block[2*i] - 'a' + 10;
        bytes[i].nibble2 = (block[2*i+1] >= '0' && block[2*i+1] <= '9') ? block[2*i+1] - '0' : block[2*i+1] - 'a' + 10;
    }
    return bytes;
}

uint64_t fromBytesToLong(byte* bytes) {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result = (result << 4) | (bytes[i].nibble1 & 0xF);
        result = (result << 4) | (bytes[i].nibble2 & 0xF);
    }
    return result;
}

uint64_t fromHexStringToLong(const string &block) {
    uint64_t result = 0;
    for (int i = 0; i < 16; i++) {
        result = (result << 4) | ((block[i] >= '0' && block[i] <= '9') ? (block[i] - '0') : (block[i] - 'a' + 10));
    }
    return result;
}

byte* fromLongToBytes(uint64_t block) {
    byte* bytes = new byte[8];
    for (int i = 7; i >= 0; --i) {
        bytes[i].nibble2 = (block >> (2 * (7 - i) * 4)) & 0xF;
        bytes[i].nibble1 = (block >> ((2 * (7 - i) + 1) * 4)) & 0xF;
    }
    return bytes;
}

string fromLongToHexString(uint64_t block) {
    stringstream ss;
    ss << hex << setfill('0') << setw(16) << block;
    return ss.str();
}

uint8_t Sbox(uint8_t input) {
    return S[input];
}

uint8_t inverseSbox(uint8_t input) {
    return invS[input];
}

uint64_t permute(uint64_t source) {
    uint64_t permutation = 0;
    for (int i = 0; i < 64; i++) {
        int distance = 63 - i;
        permutation |= ((source >> distance) & 0x1ULL) << (63 - P[i]);
    }
    return permutation;
}

uint64_t inversepermute(uint64_t source) {
    uint64_t permutation = 0;
    for (int i = 0; i < 64; i++) {
        int distance = 63 - P[i];
        permutation = (permutation << 1) | ((source >> distance) & 0x1ULL);
    }
    return permutation;
}

uint16_t getKeyLow(const string &key) {
    uint16_t keyLow = 0;
    for (int i = 16; i < 20; ++i) {
        keyLow = (keyLow << 4) | (((key[i] >= '0' && key[i] <= '9') ? key[i] - '0' : key[i] - 'a' + 10) & 0xF);
    }
    return keyLow;
}

uint64_t* generateSubkeys(const string &key) {
    uint64_t keyHigh = fromHexStringToLong(key);
    uint16_t keyLow = getKeyLow(key);
    uint64_t* subKeys = new uint64_t[32];
    subKeys[0] = keyHigh;

    for (int i = 1; i < 32; ++i) {
        uint64_t temp1 = keyHigh, temp2 = keyLow;
        keyHigh = (keyHigh << 61) | (temp2 << 45) | (temp1 >> 19);
        keyLow = (temp1 >> 3) & 0xFFFF;

        uint8_t temp = Sbox(keyHigh >> 60);
        keyHigh &= 0x0FFFFFFFFFFFFFFF;
        keyHigh |= ((uint64_t)temp) << 60;

        keyLow ^= ((i & 0x01) << 15);
        keyHigh ^= (i >> 1);

        subKeys[i] = keyHigh;
    }

    return subKeys;
}

string encrypt(int R, const string &plaintext, const string &key) {
    uint64_t* subkeys = generateSubkeys(key);
    uint64_t state = fromHexStringToLong(plaintext);

    for (int i = 0; i < R; i++) {
        state ^= subkeys[i];
        if(i != R-1)
        {
        byte* stateBytes = fromLongToBytes(state);
        for (int j = 0; j < 8; ++j) {
            stateBytes[j].nibble1 = Sbox(stateBytes[j].nibble1);
            stateBytes[j].nibble2 = Sbox(stateBytes[j].nibble2);
        }
        if(i != R-2)
        {
            state = permute(fromBytesToLong(stateBytes));
        }
        else
        {
            state = fromBytesToLong(stateBytes);
        }
        delete[] stateBytes;
        }
    }
    if(R == 31){
        state ^= subkeys[31];
    }

    delete[] subkeys;
    return fromLongToHexString(state);
}

string decrypt(const string &ciphertext, const string &key) {
    uint64_t* subkeys = generateSubkeys(key);
    uint64_t state = fromHexStringToLong(ciphertext);

    for (int i = 0; i < 31; i++) {
        state ^= subkeys[31 - i];
        state = inversepermute(state);
        byte* stateBytes = fromLongToBytes(state);
        for (int j = 0; j < 8; ++j) {
            stateBytes[j].nibble1 = inverseSbox(stateBytes[j].nibble1);
            stateBytes[j].nibble2 = inverseSbox(stateBytes[j].nibble2);
        }
        state = fromBytesToLong(stateBytes);
        delete[] stateBytes;
    }

    state ^= subkeys[0];
    delete[] subkeys;
    return fromLongToHexString(state);
}

void splitHexStringToArray(const string& hexString, uint8_t arr[16]) {
    if (hexString.length() != 16) {
        cerr << "Error: hexString must be exactly 16 characters long." << endl;
        return;
    }

    for (int i = 0; i < 16; ++i) {
        char c = hexString[i];
        if (c >= '0' && c <= '9')
            arr[i] = c - '0';
        else if (c >= 'a' && c <= 'f')
            arr[i] = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') 
            arr[i] = c - 'A' + 10;
        else {
            cerr << "Invalid hex character: " << c << endl;
            return;
        }
    }
}


std::string arrayToHexString(const uint8_t arr[16]) {
    std::string result;
    result.reserve(16);

    for (int i = 0; i < 16; ++i) {
        if (arr[i] < 10)
            result += ('0' + arr[i]);
        else if (arr[i] < 16)
            result += ('a' + (arr[i] - 10));
        else
            throw std::invalid_argument("Array element out of range (must be 0-15)");
    }

    return result;
}


void enc_package(int R,  uint8_t ciphertext_xor[16][16], int choose_block, uint8_t  num_and_key[2][16])
{



    uint8_t tk1[R][16];
    uint8_t tk2[R][16];
    uint8_t rtk[R][8];
    
    uint8_t ciphertext_1[16];
    uint8_t ciphertext_2[16];


    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 g2(seed1); 

    uint64_t plaintext_1_random = g2();
    uint64_t key_1_random = g2();
    uint64_t key_2_random = static_cast<uint16_t>(g2());
    
    std::stringstream random_all_key;
    random_all_key << std::hex << std::setfill('0') << std::setw(16) << key_1_random
       << std::setw(4) << key_2_random;

    uint8_t plaintext_1[16];
    for (uint8_t i = 0; i < 16; i++) {
        plaintext_1[i] = static_cast<uint8_t>(plaintext_1_random >> (4 * (15 - i)) & 0xF);  
    }

    string random_key = random_all_key.str();

    for( int i =0; i < 16; i ++)
    {
        plaintext_1[choose_block] = i;

        string plaintext_string = arrayToHexString(plaintext_1);
        string cipher_string = encrypt(R, plaintext_string, random_key);
        splitHexStringToArray(cipher_string , ciphertext_1);
        for(int j = 0; j < 16; j ++)
        {
            ciphertext_xor[i][j] = ciphertext_1[j];

        }
    }

}




int main() {

    int R = 10;
    int choose_block = 6;
    int train_num = 20;
    uint8_t ciphertext_no_xor[16][16]; 
    uint8_t num_and_key[2][16];

    double repeat_num = pow(2, train_num);

    std::string fileName = generateFileName(train_num,  R, choose_block);


    ofstream outfile(fileName,  ios::app);

    auto start = high_resolution_clock::now();

    int block[]= {8, 10, 11};
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
        outfile << endl;

}


    outfile.close();

    auto end = high_resolution_clock::now();

}