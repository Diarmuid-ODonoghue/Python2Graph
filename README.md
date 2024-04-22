# Python2Graph
Information extraction applied to Python source code producing *knowledge graphs*.
One graph is generated for each method, in the form of triples. 

|  node1 | relation | node2 | 
| ------ | ------ | ------ | 
| 0. Module | Contains | 1. Block | 
| 2. Block	| Contains	 | 2. FunctionDef : start | 
| 3. FunctionDef : start	 | Contains  | 3. Block | 
| 4. Block	| Contains	 | 4. Assign | 
| 4. Assign	| Contains	| 5. Name : count | 


