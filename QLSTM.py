import torch
import torch.nn as nn
import pennylane as qml


class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits=4, n_layers=1, batch_first=True, dev_type="default.qubit"):
        super(QLSTM, self).__init__()
        # 定义QLSTM的输入，h,比特数,层数,device的类型
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.input_size + self.hidden_size
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev_type = dev_type

        self.batch_first = batch_first

        # 定义四个VQC的线路名字
        self.wires_forget = [f"{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"{i}" for i in range(self.n_qubits)]

        # 定义四个node的device
        self.dev_forget = qml.device(self.dev_type, wires=self.wires_forget)
        self.dev_input = qml.device(self.dev_type, wires=self.wires_input)
        self.dev_update = qml.device(self.dev_type, wires=self.wires_update)
        self.dev_output = qml.device(self.dev_type, wires=self.wires_output)

        # 定义 forget线路
        def circiut_forget(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        self.layer_forget = qml.QNode(circiut_forget, self.dev_forget, interface="torch")

        # 定义 input线路
        def circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        self.layer_input = qml.QNode(circuit_input, self.dev_input, interface="torch")

        # 定义 update线路
        def circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        self.layer_update = qml.QNode(circuit_update, self.dev_update, interface="torch")

        # 定义output线路
        def circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        self.layer_output = qml.QNode(circuit_output, self.dev_output, interface="torch")

        weight_shape = {"weights": (n_layers, n_qubits)}
        print(f"weight_shapes = (n_layer, n_qubits) = ({n_layers}, {n_qubits})")

        self.layer_in = torch.nn.Linear(self.concat_size, n_qubits)

        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.layer_forget, weight_shape),
            'input': qml.qnn.TorchLayer(self.layer_input, weight_shape),
            'update': qml.qnn.TorchLayer(self.layer_update, weight_shape),
            'output': qml.qnn.TorchLayer(self.layer_output, weight_shape)
        }

        self.layer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):

        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, feature_size = x.size()  # 词的长度， h_t 的 shape = (batch_size, feature_size)

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]
        # 得到第t个元素的
        for t in range(seq_length):
            if self.batch_first is True:
                x_t = x[:, t, :]
            else:
                x_t = x[t, :, :]

            v_t = torch.cat((h_t, x_t), dim=1)
            y_t = self.layer_in(v_t)

            f_t = torch.sigmoid(self.layer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.layer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.layer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.layer_out(self.VQC['output'](y_t)))  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
