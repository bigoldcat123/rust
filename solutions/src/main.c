#include<stdio.h>
int main() {
    int k = 0;
    int b;
    for (int i = 1; i <= 5; i++ ) {
        b = i % 2;
        while(b-- > 0) k++;
    }
    printf("%d,%d\n",k,b);
    return 0;
}
