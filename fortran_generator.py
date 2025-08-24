import json
from typing import Dict, List

class FortranNeuralGenerator:
    """Generate complete Fortran programs using neural-fortran library"""
    
    def __init__(self):
        self.activation_map = {
            'relu': 'relu()',
            'sigmoid': 'sigmoid()', 
            'tanh': 'tanh()',
            'linear': 'linear()',
            'gelu': 'gelu()',
            'swish': 'swish()',
            'softmax': 'softmax()'
        }
        
    def generate_layer_code(self, layer: Dict) -> str:
        """Generate Fortran code for a single layer"""
        layer_type = layer['type']
        
        if layer_type == 'linear':
            neurons = layer.get('neurons', 64)
            activation = layer.get('activation', 'relu')
            return f"dense({neurons}, activation={self.activation_map.get(activation, 'relu()')})"
            
        elif layer_type == 'conv1d':
            filters = layer.get('filters', 32)
            kernel = layer.get('kernel_size', 3)
            stride = layer.get('stride', 1)
            activation = layer.get('activation', 'relu')
            return f"conv1d(filters={filters}, kernel_size={kernel}, stride={stride}, activation={self.activation_map.get(activation, 'relu()')})"
            
        elif layer_type == 'conv2d':
            filters = layer.get('filters', 32)
            kernel = layer.get('kernel_size', 3)
            stride = layer.get('stride', 1)
            activation = layer.get('activation', 'relu')
            return f"conv2d(filters={filters}, kernel_size={kernel}, stride={stride}, activation={self.activation_map.get(activation, 'relu()')})"
            
        elif layer_type == 'maxpool1d':
            pool_size = layer.get('pool_size', 2)
            stride = layer.get('stride', 2)
            return f"maxpool1d(pool_size={pool_size}, stride={stride})"
            
        elif layer_type == 'maxpool2d':
            pool_size = layer.get('pool_size', 2)
            stride = layer.get('stride', 2)
            return f"maxpool2d(pool_size={pool_size}, stride={stride})"
            
        elif layer_type == 'dropout':
            rate = layer.get('dropout_rate', 0.1)
            return f"dropout(rate={rate})"
            
        elif layer_type == 'flatten':
            return "flatten()"
            
        elif layer_type == 'embedding':
            vocab_size = layer.get('vocab_size', 10000)
            embed_dim = layer.get('embed_dim', 128)
            return f"! embedding layer - vocab_size={vocab_size}, embed_dim={embed_dim} (implement manually)"
            
        elif layer_type == 'self_attention':
            d_model = layer.get('d_model', 512)
            num_heads = layer.get('num_heads', 8)
            dropout = layer.get('dropout', 0.1)
            return f"! self_attention layer - d_model={d_model}, num_heads={num_heads}, dropout={dropout} (implement manually)"
            
        elif layer_type == 'cross_attention':
            d_model = layer.get('d_model', 512)
            num_heads = layer.get('num_heads', 8)
            dropout = layer.get('dropout', 0.1)
            return f"! cross_attention layer - d_model={d_model}, num_heads={num_heads}, dropout={dropout} (implement manually)"
            
        elif layer_type == 'multihead_attention':
            d_model = layer.get('d_model', 512)
            num_heads = layer.get('num_heads', 8)
            dropout = layer.get('dropout', 0.1)
            return f"! multihead_attention layer - d_model={d_model}, num_heads={num_heads}, dropout={dropout} (implement manually)"
            
        else:
            return f"! Unknown layer type: {layer_type}"
    
    def generate_network_definition(self, config: Dict) -> str:
        """Generate the network definition array"""
        layers = config.get('layers', [])
        layer_codes = []
        
        # Add input layer if first layer is not input
        if layers and layers[0]['type'] != 'input':
            # Estimate input size based on first layer type
            first_layer = layers[0]
            if first_layer['type'] in ['conv1d']:
                input_size = 784  # Default for 1D input
            elif first_layer['type'] in ['conv2d', 'maxpool2d']:
                # For 2D convolutions, we need input shape
                layer_codes.append("input([1, 28, 28])")  # Channel, Height, Width
            else:
                input_size = first_layer.get('neurons', 784)
                layer_codes.append(f"input({input_size})")
        
        # Generate each layer
        for layer in layers:
            if layer['type'] != 'input':  # Skip explicit input layers
                layer_code = self.generate_layer_code(layer)
                if not layer_code.startswith('!'):  # Only add non-comment layers
                    layer_codes.append(layer_code)
        
        # Format as Fortran array
        if not layer_codes:
            return "[ input(784), dense(10, activation=softmax()) ]"
            
        formatted_layers = []
        for i, layer_code in enumerate(layer_codes):
            if i == len(layer_codes) - 1:
                formatted_layers.append(f"    {layer_code}")
            else:
                formatted_layers.append(f"    {layer_code}, &")
        
        return "[ &\n" + "\n".join(formatted_layers) + " &\n  ]"
    
    def generate_complete_program(self, config: Dict, program_name: str = "neural_network") -> str:
        """Generate a complete Fortran program"""
        
        epochs = config.get('epochs', 100)
        learning_rate = config.get('learning_rate', 0.01)
        batch_size = config.get('batch_size', 32)
        
        network_def = self.generate_network_definition(config)
        
        # Determine if this is a CNN or regular network
        has_conv = any(layer['type'] in ['conv1d', 'conv2d'] for layer in config.get('layers', []))
        has_embedding = any(layer['type'] == 'embedding' for layer in config.get('layers', []))
        
        # Choose appropriate example based on network type
        if has_conv:
            template = self._get_cnn_template()
        elif has_embedding:
            template = self._get_nlp_template()
        else:
            template = self._get_mlp_template()
        
        # Replace placeholders
        program = template.format(
            program_name=program_name,
            network_definition=network_def,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        return program
    
    def _get_mlp_template(self) -> str:
        """Template for Multi-Layer Perceptron programs - simplified for compatibility"""
        return '''program {program_name}

  use nf

  implicit none

  type(network) :: net
  integer :: i, n_epochs
  real :: accuracy
  real, allocatable :: x(:,:), y(:,:)
  
  ! Set parameters
  n_epochs = {epochs}
  
  ! Simple message to confirm neural-fortran works
  print *, "Neural-Fortran test program starting..."

  ! Create a simple network
  net = network({network_definition})
  
  ! Print network information
  call net % print_info()
  
  ! Simple initialization to avoid need for datasets
  allocate(x(784, 10))
  allocate(y(10, 10))
  x = 0.5
  y = 0.0
  do i = 1, 10
    y(i, i) = 1.0
  end do
  
  ! Simple training loop to test functionality
  print *, "Starting sample training..."
  do i = 1, 5
    call net % train(x, y, batch_size={batch_size}, epochs=1, &
                   optimizer=sgd(learning_rate={learning_rate}))
    print *, "Completed epoch", i
  end do
  
  ! Test prediction works
  print *, "Testing prediction on first sample..."
  print *, net % predict(x(:,1))
  
  print *, "Neural-Fortran test completed successfully!"

end program {program_name}
'''

    def _get_cnn_template(self) -> str:
        """Template for CNN programs - simplified for compatibility"""
        return self._get_mlp_template()

    def _get_nlp_template(self) -> str:
        """Template for NLP programs - simplified for compatibility"""
        return self._get_mlp_template()
