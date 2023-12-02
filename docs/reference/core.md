
## Eager execution
When you call something like `x+1` and `slope.zeros(3)` global namespace, Slope 
1. Generates program intermediate representation IR (`slope.core.Program`)
2. Translate this IR to backend code codegen
3. Compile this code with the backend
4. Execute the code.

Jitting operations one-by-one can be considered eager execution, hence is slow

## Lazy execution

Like JAX, the good practice is to write blocks of code as functions, then use `slope.jit` decorator.
