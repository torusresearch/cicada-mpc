Feature: Arithmetic

    Scenario Outline: Field Array
        Given a <field>
	And a field array <a>
	Then the field array should match <b>

        Examples:
        | field                     | a                  | b                  |
        | Field with default order  | 1                  | 1                  |
        | Field with default order  | [1]                | [1]                |
        | Field with default order  | [1,2,3]            | [1,2,3]            |
        | Field with default order  | [[1,2,3],[4,5,6]]  | [[1,2,3],[4,5,6]]  |
        | Field with order 251      | -1                 | 250                |
        | Field with order 251      | 252                | 1                  |


    Scenario Outline: Field Array Addition
	Given a <field>
	And a field array <a>
	And a field array <b>
	When the second field array is added to the first
	Then the field array should match <c>

        Examples:
        | field                     | a         | b         | c        |
        | Field with default order  | 1         | 1         | 2        |
        | Field with default order  | 1         | 3         | 4        |
        | Field with default order  | 1         | 0         | 1        |
        | Field with default order  | 0         | 1         | 1        |
        | Field with default order  | -1        | 1         | 0        |


    Scenario Outline: Field Array In-Place Addition
	Given a <field>
	And a field array <a>
	And a field array <b>
	When the second field array is added in-place to the first
	Then the field array should match <c>

        Examples:
        | field                     | a         | b         | c        |
        | Field with default order  | 1         | 1         | 2        |
        | Field with default order  | 1         | 3         | 4        |
        | Field with default order  | 1         | 0         | 1        |
        | Field with default order  | 0         | 1         | 1        |
        | Field with default order  | -1        | 1         | 0        |


    Scenario Outline: Field Array In-Place Subtraction
	Given a <field>
	And a field array <a>
	And a field array <b>
	When the second field array is subtracted in-place from the first
	Then the field array should match <c>

        Examples:
        | field                     | a         | b         | c        |
        | Field with default order  | 1         | 1         | 0        |
        | Field with default order  | 1         | 3         | -2       |
        | Field with default order  | 1         | 0         | 1        |
        | Field with default order  | 0         | 1         | -1       |


    Scenario Outline: Field Array Negation
	Given a <field>
	And a field array <a>
	When the field array is negated
	Then the field array should match <b>

        Examples:
        | field                     | a         | b                  |
        | Field with order 251      | 0         | 0                  |
        | Field with order 251      | 1         | 250                |
        | Field with order 251      | 3         | 248                |
        | Field with order 251      | -1        | 1                  |
        | Field with order 251      | -3        | 3                  |
        | Field with order 251      | [1, 2]    | [250, 249]         |


    Scenario Outline: Field Array Subtraction
	Given a <field>
	And a field array <a>
	And a field array <b>
	When the second field array is subtracted from the first
	Then the field array should match <c>

        Examples:
        | field                     | a         | b         | c        |
        | Field with default order  | 1         | 1         | 0        |
        | Field with default order  | 1         | 3         | -2       |
        | Field with default order  | 1         | 0         | 1        |
        | Field with default order  | 0         | 1         | -1       |


    Scenario Outline: Field Array Summation
	Given a <field>
	And a field array <a>
	When the field array is summed
	Then the field array should match <b>

        Examples:
        | field                     | a         | b         |
	| Field with default order  | [1, 2, 3] | 6         |


    Scenario Outline: Field Equality
        Given a <lhs>
        And a <rhs>
        Then the fields should compare <result>

        Examples:
        | lhs                       | rhs                              | result   |
        | Field with default order  | Field with default order         | equal    |
        | Field with default order  | Field with order 251             | unequal  |


    Scenario Outline: Field Zeros
        Given a <field>
	When generating a field array of zeros with shape <shape>
	Then the field array should match <result>

        Examples:
        | field                           | shape              | result                |
        | Field with default order        | ()                 | 0                     |
        | Field with default order        | 1                  | [0]                   |
        | Field with default order        | (2, 3)             | [[0, 0, 0],[0, 0, 0]] |


    Scenario Outline: Field Zeros Like
        Given a <field>
        When generating a field array of zeros like <other>
	Then the field array should match <result>

        Examples:
        | field                           | other              | result                |
        | field with default order        | 3                  | 0                     |
        | field with default order        | [3, 4]             | [0, 0]                |
        | field with default order        | [[3,4,5],[5,6,7]]  | [[0, 0, 0],[0, 0, 0]] |


