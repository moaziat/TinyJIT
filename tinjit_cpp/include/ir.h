#ifndef IR_H 
#define IR_H

#include <string> 
#include <vector> 

enum DataType { 
    FLOAT32
}; 

class Tensor; 
class Op; 
class Function; 

class Tensor {
public: 
    std::string name; 
    std::vector<int> shape; 
    DataType dtype; 
    Op* op; 


    static int counter; 
    Tensor(const std::vector<int>& shape, DataType dtype = FLOAT32, Op* op = nullptr, const std::string& name = ""); 
    Tensor* add(Tensor* other); 
    std::string toString() const;
}; 

class Op {
public: 
    std::string op_type; 
    std::vector<Tensor*> inputs; 
    std::vector<Tensor*> outputs; 
    std::string name; 

    static int counter; 
    Op(const std::string& op_type, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    std::string toString() const;
};


class Function {
public: 
    std::string name; 
    std::vector<Tensor*> outputs; 
    std::vector<Op*> ops; 

    Function(const std::string& name, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);
    std::string toString() const;

private: 
    void collectOps(std::vector<Op*>& ordered_ops, Tensor* tensor);  // Build op list

}; 

#endif 