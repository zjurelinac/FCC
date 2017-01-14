int main(){
    char *UART_WRITE = 0xFFFF1000;
    char *UART_READ = 0xFFFF1004;
    char *UART_STAT = 0xFFFF1008;
    *UART_WRITE = 'a';
}
