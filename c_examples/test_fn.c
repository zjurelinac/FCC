int abs(int x){
    if(x < 0) return -x;
    return x;
}

int main(void){
    int a = 7, b = 4;
    printf("%d\n", abs((b-5)/a));
    return 0;
}
