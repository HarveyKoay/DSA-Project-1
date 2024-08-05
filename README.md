# DSA-Project-1
This repository contains the implementation of Data Structure and algorithm project. It focuses on problem-solving strategies, algorithmic paradigms, and their applications in solving new problems. The main objectives include proving the correctness of programs, analyzing their space and time complexities, and comparing various abstract data types.

Learning Outcomes
The project achieves the following learning outcomes:

Analyze general problem-solving strategies and algorithmic paradigms and apply them to new problems.
Prove the correctness of programs and analyze their space and time complexities.
Compare and contrast various abstract data types and use them appropriately.
Develop and implement algorithms to solve computational problems.
Additionally, the project helps in developing the following employability skills:

Text comprehension.
Designing test cases.
Ability to follow specifications precisely.
Project Structure
Project_final.py: This is the main file containing all the solutions for the assignment.
README.md: This file, providing an overview of the project and its structure.
Implementation Details
Question 1: Ultimate Fuse
Problem Description
In the FITWORLD, you need to fuse FITMONs to create the cutest FITMON. Each FITMON has a cuteness score and affinities for fusing with adjacent FITMONs. The objective is to fuse all given FITMONs into one ultimate FITMON with the highest possible cuteness score.

Input
A list of FITMONs, each represented as [affinity_left, cuteness_score, affinity_right].
Output
The cuteness score of the ultimate FITMON after fusing all given FITMONs optimally.
Complexity
The fuse(fitmons) function is designed to run with a worst-case time complexity of O(N^3) and space complexity of O(N^2).
Example
python
Copy code
fitmons = [
    [0, 29, 0.9],
    [0.9, 91, 0.8],
    [0.8, 48, 0]
]
print(fuse(fitmons))  # Output: 126

Question 2: Delulu is not the Solulu
Problem Description
You need to navigate through the Delulu Forest to escape. The forest is represented as a graph where trees are nodes and roads are edges. The task is to find the shortest path to an exit tree, considering that some exits might be delusions.

Input
A list of trees, each represented as nodes in a graph.
A list of roads, each represented as edges with weights.
Output
The shortest time to escape the forest or determine if it's impossible.
Complexity
The function for solving this problem is designed to handle large inputs efficiently, utilizing appropriate graph traversal algorithms.

This repository contains the implementation of Assignment 2 for the FIT2004 unit in the first semester of 2024. The assignment includes solving problems related to open reading frames in DNA sequences and optimizing the assignment of security officers to companies.

# DSA-Project-2
Analyze general problem-solving strategies and algorithmic paradigms and apply them to solving new problems.
Prove correctness of programs and analyze their space and time complexities.
Compare and contrast various abstract data types and use them appropriately.
Develop and implement algorithms to solve computational problems.
Additionally, the project helps in developing the following employability skills:

Text comprehension.
Designing test cases.
Ability to follow specifications precisely.
Project Structure
DSA Project 2.py: This is the main file containing all the solutions for the assignment.
README.md: This file, providing an overview of the project and its structure.
Implementation Details
Question 1: Open Reading Frames
Problem Description
In molecular genetics, an Open Reading Frame (ORF) is a portion of DNA used as the blueprint for a protein. The task is to find all sections of a genome that start with a given sequence and end with a (possibly) different given sequence.

Input
genome: A single non-empty string consisting only of uppercase [A-D].
start: A single non-empty string consisting only of uppercase [A-D].
end: A single non-empty string consisting only of uppercase [A-D].
Output
A list of strings containing all substrings of the genome that have start as a prefix and end as a suffix.
Complexity
The __init__ method of OrfFinder must run in O(N²) time complexity, where N is the length of the genome.
The find method must run in O(T + U + V) time complexity, where T is the length of the start string, U is the length of the end string, and V is the number of characters in the output list.
Example
python
Copy code
genome1 = OrfFinder("AAABBBCCC")
print(genome1.find("AAA", "BB"))  # Output: ["AAABB", "AAABBB"]
Question 2: Securing the Companies
Problem Description
As a manager of a security company, the task is to assign security officers to companies for a month, ensuring each company gets the required number of officers for each shift per day. Constraints include officers' shift preferences and the total number of shifts an officer can work in a month.

Input
preferences: A list of lists where preferences[i][k] is a binary value indicating if officer i is interested in shift k.
officers_per_org: A list of lists where officers_per_org[j][k] specifies how many officers company j needs for shift k each day.
Output
None if no allocation satisfying all constraints exists.
Otherwise, a list of lists allocation where allocation[i][j][d][k] is 1 if officer i is allocated to company j for shift k on day d.
Complexity
The solution must have a worst-case time complexity of O(m * n * n).
