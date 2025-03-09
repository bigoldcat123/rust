fn main() {}

pub fn is_valid(code: &str) -> bool {
    todo!("Is the Luhn checksum for {code} valid?");
}

/*
 * Validating a number
* * Strings of length 1 or less are not valid. Spaces are allowed in the input, but they should be stripped before checking. All other non-digit characters are disallowed.
* *
* * Example 1: valid credit card number
* * 4539 3195 0343 6467
* * The first step of the Luhn algorithm is to double every second digit, starting from the right. We will be doubling
* *
* * 4539 3195 0343 6467
* * ↑ ↑  ↑ ↑  ↑ ↑  ↑ ↑  (double these)
* * If doubling the number results in a number greater than 9 then subtract 9 from the product. The results of our doubling:
* *
* * 8569 6195 0383 3437
* * Then sum all of the digits:
* *
* * 8+5+6+9+6+1+9+5+0+3+8+3+3+4+3+7 = 80
* * If the sum is evenly divisible by 10, then the number is valid. This number is valid!
 */
