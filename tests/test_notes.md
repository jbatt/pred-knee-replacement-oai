Integration tests are more robust when they focus on general high level outcomes that we don’t expect to change often. Integration tests that check very specific outcomes will need to be updated with any small change to the logic within the part that is being tested.


Each time you find a bug in your code, you should write a new test to assert that the code works correctly. Once the bug is fixed, this new test should pass and give you confidence that the bug has been fixed.

Test-driven development typically repeats three steps:

    Red - Write a test that we expect to fail

    Green - Write or update our code to pass the new test

    Refactor - Make improvements to the quality of the code without changing the functionality
