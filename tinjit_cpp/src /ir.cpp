#include "../include/ir.h"
#include <sstream>  // For std::stringstream

int Tensor::counter = 0;
Tensor::Tensor(const std::vector<int>& shape, DataType dtype, Op* op, const std::string& name)
    : shape(shape), dtype(dtype), op(op) {
    if (name.empty()) {
        this->name = "t" + std::to_string(counter++);
    } else {
        this->name = name;
    }
}

Tensor* Tensor::add(Tensor* other) {
    // Check if shapes match (e.g., both 2x2)
    if (shape != other->shape) {
        throw std::runtime_error("Shape mismatch for add");
    }
    // Create output tensor with same shape
    Tensor* out = new Tensor(shape, dtype);
    // Create add operation
    Op* op = new Op("add", {this, other}, {out});
    out->op = op;
    return out;
}

std::string Tensor::toString() const {
    std::stringstream ss;
    ss << "%" << name << ": float32(";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i < shape.size() - 1) ss << ",";
    }
    ss << ")";
    return ss.str();
}

int Op::counter = 0;
Op::Op(const std::string& op_type, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : op_type(op_type), inputs(inputs), outputs(outputs) {
    name = "op" + std::to_string(counter++);
}

std::string Op::toString() const {
    std::stringstream ss;
    ss << name << ": ";
    for (const auto& out : outputs) ss << out->name << " ";
    ss << "= " << op_type << "(";
    for (const auto& in : inputs) ss << in->name << " ";
    ss << ")";
    return ss.str();
}

Function::Function(const std::string& name, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
    : name(name), args(inputs), outputs(outputs) {
    std::vector<Op*> ordered_ops;
    for (const auto& out : outputs) {
        collectOps(ordered_ops, out);
    }
    ops = ordered_ops;
}

void Function::collectOps(std::vector<Op*>& ordered_ops, Tensor* tensor) {
    if (!tensor->op || std::find(ordered_ops.begin(), ordered_ops.end(), tensor->op) != ordered_ops.end()) {
        return;
    }
    for (const auto& inp : tensor->op->inputs) {
        collectOps(ordered_ops, inp);
    }
    ordered_ops.push_back(tensor->op);
}

std::string Function::toString() const {
    std::stringstream ss;
    ss << "func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
        ss << args[i]->toString();
        if (i < args.size() - 1) ss << ", ";
    }
    ss << ") {\n";
    for (const auto& op : ops) ss << "  " << op->toString() << "\n";
    ss << "  return ";
    for (size_t i = 0; i < outputs.size(); ++i) {
        ss << "%" << outputs[i]->name;
        if (i < outputs.size() - 1) ss << ", ";
    }
    ss << "\n}";
    return ss.str();
}