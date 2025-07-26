# stable_diffusion_inference.py
import onnxruntime_genai as og
import numpy as np
from PIL import Image
import argparse
import os
import time

def run_stable_diffusion_inference(model_path, prompt, steps=20, guidance_scale=7.5, execution_provider="OpenVINO"):
    """Run Stable Diffusion inference using ONNX Runtime GenAI with configurable execution provider"""
    
    try:

        provider_map = {
            "openvino": "OpenVINO"
        }
        actual_provider = provider_map.get(execution_provider, execution_provider)

        # Create config and configure execution provider
        print(f"Configuring execution provider: {execution_provider}")
        config = og.Config(model_path)
        
        # Clear default providers and add the specified one
        config.clear_providers()
        
        if execution_provider.lower() != "cpu":
            config.append_provider(actual_provider)
            
            # Configure OpenVINO-specific options
            if execution_provider.lower() == "openvino":
                config.set_provider_option(actual_provider, "device_type", "GPU")
                #config.set_provider_option(actual_provider, "enable_dynamic_shapes", "1")
                config.set_provider_option(actual_provider, "num_of_threads", "0")  # Auto thread count
                print("✅ OpenVINO provider configured with CPU device and dynamic shapes")
            
            # Configure CUDA-specific options
            elif execution_provider.lower() == "cuda":
                config.set_provider_option("cuda", "enable_cuda_graph", "0")
                print("✅ CUDA provider configured")
        else:
            print("✅ Using CPU provider")
        
        # Load the model with the configured execution provider
        print(f"Loading model from: {model_path}")
        model = og.Model(config)  # Pass config to model constructor
        
        # Create image generator parameters
        params = og.ImageGeneratorParams(model)
        
        # Set the prompt
        params.set_prompt(prompt)
        
        print(f"Generating image for prompt: '{prompt}' using {execution_provider}")
        
        # Measure generation time
        start_time = time.time()
        
        # Generate image
        image_tensor = og.generate_image(model, params)
        
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"⏱️  Image generation completed in {generation_time:.2f} seconds")
        
        # Convert tensor to numpy array (following standalone test approach)
        image_array = image_tensor.as_numpy()
        
        print(f"Generated image tensor shape: {image_array.shape}")
        print(f"Generated image tensor dtype: {image_array.dtype}")
        
        # Handle batch dimension - create list of PIL images (following standalone test)
        if len(image_array.shape) == 4:
            # Multiple images in batch
            images = [Image.fromarray(image_array[i]) for i in range(image_array.shape[0])]
        else:
            # Single image
            images = [Image.fromarray(image_array)]
        
        # Save all generated images
        saved_paths = []
        for i, image in enumerate(images):
            # Create safe filename with execution provider info
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:30]  # Limit length
            
            if len(images) > 1:
                output_path = f"generated_image_{safe_prompt}_{execution_provider}_{i}.png"
            else:
                output_path = f"generated_image_{safe_prompt}_{execution_provider}.png"
            
            image.save(output_path)
            saved_paths.append(output_path)
            print(f"Image {i+1} saved to: {output_path}")
        
        print(f"🎉 Successfully generated {len(images)} image(s) using {execution_provider} in {generation_time:.2f}s")
        return images
        
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        print(f"Model path exists: {os.path.exists(model_path)}")
        
        # Debug information
        try:
            config = og.Config(model_path)
            model = og.Model(config)
            params = og.ImageGeneratorParams(model)
            print(f"Model type: {model.type}")
            print(f"Available params methods: {[m for m in dir(params) if not m.startswith('_')]}")
        except Exception as debug_e:
            print(f"Debug error: {debug_e}")
        
        return None

def benchmark_execution_providers(model_path, prompt, providers=["cpu", "openvino"]):
    """Benchmark different execution providers"""
    print("=" * 60)
    print("🏁 EXECUTION PROVIDER BENCHMARK")
    print("=" * 60)
    
    results = {}
    
    for provider in providers:
        print(f"\n🔄 Testing {provider.upper()} execution provider...")
        
        try:
            start_time = time.time()
            images = run_stable_diffusion_inference(model_path, prompt, execution_provider=provider)
            end_time = time.time()
            
            if images:
                results[provider] = end_time - start_time
                print(f"✅ {provider.upper()}: {results[provider]:.2f} seconds")
            else:
                results[provider] = None
                print(f"❌ {provider.upper()}: Failed")
                
        except Exception as e:
            results[provider] = None
            print(f"❌ {provider.upper()}: Error - {e}")
    
    # Print benchmark results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if successful_results:
        fastest_provider = min(successful_results, key=successful_results.get)
        
        for provider, time_taken in successful_results.items():
            if provider == fastest_provider:
                print(f"🥇 {provider.upper()}: {time_taken:.2f}s (FASTEST)")
            else:
                speedup = successful_results[fastest_provider] / time_taken
                print(f"   {provider.upper()}: {time_taken:.2f}s ({speedup:.2f}x slower)")
        
        if len(successful_results) > 1:
            best_time = successful_results[fastest_provider]
            worst_time = max(successful_results.values())
            speedup = worst_time / best_time
            print(f"\n🚀 Best speedup: {speedup:.2f}x ({fastest_provider.upper()} vs others)")
    else:
        print("❌ No execution providers worked successfully")

def test_model_compatibility(model_path, execution_provider="openvino"):
    """Test model compatibility with specified execution provider"""
    print(f"=== Model Compatibility Test ({execution_provider.upper()}) ===")
    
    try:
        # Create config with execution provider
        config = og.Config(model_path)
        config.clear_providers()
        
        if execution_provider.lower() != "cpu":
            config.append_provider(execution_provider)
            if execution_provider.lower() == "openvino":
                config.set_provider_option("openvino", "device_type", "CPU")
                config.set_provider_option("openvino", "enable_dynamic_shapes", "1")
        
        # Test model loading
        model = og.Model(config)
        print(f"✅ Model loaded successfully with {execution_provider}")
        print(f"   Model type: {model.type}")
        print(f"   Device type: {model.device_type}")
        
        # Test parameters creation
        params = og.ImageGeneratorParams(model)
        print(f"✅ ImageGeneratorParams created successfully")
        
        # Test prompt setting
        test_prompt = "test prompt"
        params.set_prompt(test_prompt)
        print(f"✅ Prompt setting works")
        
        # Test tokenizer if available
        try:
            tokenizer = og.Tokenizer(model)
            test_tokens = tokenizer.encode(test_prompt)
            print(f"✅ Tokenizer works - shape: {test_tokens.shape}, dtype: {test_tokens.dtype}")
        except Exception as e:
            print(f"⚠️  Tokenizer test failed: {e}")
        
        print(f"✅ All compatibility tests passed with {execution_provider}!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed with {execution_provider}: {e}")
        return False

def investigate_api(model_path, execution_provider="openvino"):
    """Investigate available API methods with specified execution provider"""
    print(f"=== Enhanced API Investigation ({execution_provider.upper()}) ===")
    
    try:

        provider_map = {
                "openvino": "OpenVINO"
            }
        actual_provider = provider_map.get(args.execution_provider, args.execution_provider)

        # Create config with execution provider
        config = og.Config(model_path)
        config.clear_providers()
        
        if execution_provider.lower() != "cpu":
            config.append_provider(actual_provider)

            if execution_provider.lower() == "openvino":
                config.set_provider_option("OpenVINO", "device_type", "CPU")
                config.set_provider_option("OpenVINO", "num_of_threads", "0")
                #config.set_provider_option("openvino", "enable_dynamic_shapes", "1")

        model = og.Model(config)
        print(f"Model methods: {[m for m in dir(model) if not m.startswith('_')]}")
        
        params = og.ImageGeneratorParams(model)
        print(f"ImageGeneratorParams methods: {[m for m in dir(params) if not m.startswith('_')]}")
        
        # Test a simple generation to see tensor properties
        params.set_prompt("a simple test")
        
        print(f"\nTesting image generation with {execution_provider}...")
        start_time = time.time()
        image_tensor = og.generate_image(model, params)
        end_time = time.time()
        
        image_array = image_tensor.as_numpy()
        
        print(f"\nGenerated tensor properties:")
        print(f"  Shape: {image_array.shape}")
        print(f"  Dtype: {image_array.dtype}")
        print(f"  Min value: {image_array.min()}")
        print(f"  Max value: {image_array.max()}")
        print(f"  Value range suggests: {'uint8 [0-255]' if image_array.max() > 1 else 'float [0-1] or [-1,1]'}")
        print(f"  Generation time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion with ONNX Runtime GenAI")
    parser.add_argument("-m", "--model", required=True, help="Path to the model directory")
    parser.add_argument("-p", "--prompt", help="Text prompt for image generation")
    parser.add_argument("-e", "--execution_provider", default="openvino", 
                        choices=["cpu", "openvino", "cuda", "dml"], 
                        help="Execution provider to use (default: openvino)")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of denoising steps (not implemented yet)")
    parser.add_argument("-g", "--guidance", type=float, default=7.5, help="Guidance scale (not implemented yet)")
    parser.add_argument("--test", action="store_true", help="Test model compatibility")
    parser.add_argument("--investigate", action="store_true", help="Investigate available API methods")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark different execution providers")
    
    args = parser.parse_args()
    
    if args.test:
        test_model_compatibility(args.model, args.execution_provider)
    elif args.investigate:
        investigate_api(args.model, args.execution_provider)
    elif args.benchmark:
        if args.prompt:
            benchmark_execution_providers(args.model, args.prompt)
        else:
            benchmark_execution_providers(args.model, "a dog running in the park")
    elif args.prompt:
        run_stable_diffusion_inference(args.model, args.prompt, args.steps, args.guidance, args.execution_provider)
    else:
        print("Please provide a prompt with -p, or use --test, --investigate, or --benchmark")
        print(f"Default execution provider: {args.execution_provider}")
        parser.print_help()
