#include <bits/stdc++.h>
using namespace std;

// Palindrome
bool isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int reverse = 0;
        int num = x;
        while (num > 0) {
            int d = num % 10;
            if ((reverse > INT_MAX / 10) || (reverse < INT_MIN / 10)) {
                return false;
            }
            reverse = reverse * 10 + d;
            num = num / 10;
        }
        return reverse == x;
    }

// Palindrome recursion way

int main()
{
    int n;
    cin >> n;
    cout << isPalindrome(n) << endl;
    return 0;
}