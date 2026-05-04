#!/usr/bin/env python3
"""
Script to fix the black_box_pixel_attacks.ipynb notebook
- Strengthens attack parameters
- Fixes misleading documentation
- Adds proper success metrics
"""

import json
import re

# Read the notebook
with open('black_box_pixel_attacks.ipynb', 'r') as f:
    nb = json.load(f)

# Fix 1: Update Attack 1 parameters (cell index 12)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Fix Attack 1 parameters
        if 'attack1_result = multi_pixel_attack(' in source and 'k_pixels=5' in source:
            new_source = source.replace(
                'attack1_result = multi_pixel_attack(\n    model=model,\n    img=processed_img,\n    k_pixels=5,\n    n_iterations=150,\n    population_size=8,\n    verbose=True\n)',
                'attack1_result = multi_pixel_attack(\n    model=model,\n    img=processed_img,\n    k_pixels=30,          # Increased from 5 for stronger attack\n    n_iterations=500,     # Increased from 150 for better optimization\n    population_size=20,   # Increased from 8 for better exploration\n    verbose=True\n)'
            )
            cell['source'] = new_source.split('\n')
            print("✓ Fixed Attack 1 parameters")
        
        # Fix Attack 2 parameters
        if 'attack2_result = targeted_class_attack(' in source and 'k_pixels=6' in source:
            new_source = source.replace(
                'attack2_result = targeted_class_attack(\n    model=model,\n    img=processed_img,\n    target_class=TARGET_CLASS,\n    k_pixels=6,\n    n_iterations=200,\n    population_size=8,\n    verbose=True\n)',
                'attack2_result = targeted_class_attack(\n    model=model,\n    img=processed_img,\n    target_class=TARGET_CLASS,\n    k_pixels=30,          # Increased from 6 for stronger attack\n    n_iterations=500,     # Increased from 200 for better optimization\n    population_size=20,   # Increased from 8 for better exploration\n    verbose=True\n)'
            )
            cell['source'] = new_source.split('\n')
            print("✓ Fixed Attack 2 parameters")
        
        # Fix Attack 3 parameters
        if 'attack3_result = no_detection_attack(' in source and 'k_pixels=8' in source:
            new_source = source.replace(
                'attack3_result = no_detection_attack(\n    model=model,\n    img=processed_img,\n    k_pixels=8,\n    n_iterations=250,\n    population_size=10,\n    verbose=True\n)',
                'attack3_result = no_detection_attack(\n    model=model,\n    img=processed_img,\n    k_pixels=40,          # Increased from 8 for stronger attack\n    n_iterations=800,     # Increased from 250 for better optimization\n    population_size=25,   # Increased from 10 for better exploration\n    verbose=True\n)'
            )
            cell['source'] = new_source.split('\n')
            print("✓ Fixed Attack 3 parameters")

# Fix 2: Update misleading documentation for Attack 1
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Attack 1: Multi-Pixel Confidence Reduction' in source and 'What changed in predictions:' in source:
            old_text = '''What changed in predictions:
- The attack successfully reduced detection confidences by perturbing only 5 pixels
- Compare original vs adversarial detection confidence scores

Why the attack works:
- YOLO's detection head is sensitive to localized pixel changes
- By optimizing pixel values in specific locations, we disrupt feature extraction
- Black-box optimization finds high-impact pixels without gradient information
- Small pixel changes accumulate to significantly alter detection scores'''
            
            new_text = '''What changed in predictions:
- The attack attempts to reduce detection confidences by perturbing pixels
- With strengthened parameters (30 pixels, 500 iterations), better reduction expected
- Compare original vs adversarial detection confidence scores

Attack effectiveness:
- Black-box pixel attacks require many iterations to find optimal perturbations
- More pixels and iterations increase probability of significant confidence reduction
- Success measured by: (baseline_fitness - final_fitness) / baseline_fitness * 100%
- Full success = no detections or confidences below threshold (~0.25)'''
            
            new_source = source.replace(old_text, new_text)
            if new_source != source:
                cell['source'] = new_source.split('\n')
                print("✓ Fixed Attack 1 documentation")

# Fix 3: Update misleading documentation for Attack 2
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Attack 2: Targeted Class Attack' in source and 'What changed in predictions:' in source:
            old_text = '''What changed in predictions:
- Original detections were classified according to their true aircraft type
- After the attack, detections are forced toward class {TARGET_CLASS} ({target_name})
- The attack optimizes pixels to trigger {target_name} class features in the network

Why the attack works:
- YOLO's classification head relies on high-level features from the backbone
- Strategic pixel perturbations can activate specific class-related neurons
- By maximizing target class probability and penalizing others, the attack
  creates patterns that the model associates with the target class
- This is a misclassification attack - the model is fooled about WHAT is detected'''
            
            new_text = '''What changed in predictions:
- Original detections are classified according to their true aircraft type
- Attack attempts to force detections toward class {TARGET_CLASS} ({target_name})
- Success = any detections with class label {TARGET_CLASS} and high confidence

Attack effectiveness:
- Targeted class attacks are the HARDEST objective for black-box pixel attacks
- Requires finding pixel patterns that activate specific class neurons
- Success metrics:
  * Number of detections with target class (should be > 0)
  * Confidence of target class detections (higher = better)
  * Suppression of non-target classes (fewer wrong classes = better)
- NOTE: May fail if target class features are too different from image content'''
            
            new_source = source.replace(old_text, new_text)
            if new_source != source:
                cell['source'] = new_source.split('\n')
                print("✓ Fixed Attack 2 documentation")

# Fix 4: Update misleading documentation for Attack 3
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Attack 3: No-Detection Attack' in source and 'What changed in predictions:' in source:
            old_text = '''What changed in predictions:
- Original image contains detected aircraft with bounding boxes
- Adversarial image shows NO detections (or significantly reduced)
- The model completely fails to detect any objects

Why the attack works:
- This is the strongest adversarial objective - total evasion
- By perturbing strategic pixels, we disrupt early feature extraction
- The backbone network cannot form meaningful feature representations
- Detection confidence drops below the model's threshold (typically 0.25)
- This demonstrates complete vulnerability to black-box pixel attacks
- Only 8 pixels modified, yet the model is completely blinded'''
            
            new_text = '''What changed in predictions:
- Original image contains detected aircraft with bounding boxes
- Adversarial image aims to show NO detections (complete evasion)
- Success = zero detections OR all confidences below threshold (typically 0.25)

Attack effectiveness:
- No-detection is the most desirable adversarial outcome (total evasion)
- Requires disrupting feature extraction enough to prevent any valid detections
- Success metrics:
  * Number of detections (0 = perfect, fewer = better)
  * Maximum confidence (below 0.25 = undetectable by default threshold)
  * Fitness score (0.0 = perfect attack)
- With 40 pixels and 800 iterations, much higher success probability'''
            
            new_source = source.replace(old_text, new_text)
            if new_source != source:
                cell['source'] = new_source.split('\n')
                print("✓ Fixed Attack 3 documentation")

# Fix 5: Update summary table to be more accurate
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'Attack Effectiveness Summary' in source and 'Attack 1' in source:
            old_table = '''| Attack | Objective | Pixels Used | Effectiveness |
|--------|-----------|-------------|-----------------|
| **Attack 1** | Confidence Reduction | 5 | Reduces detection confidence |
| **Attack 2** | Targeted Class | 6 | Forces misclassification |
| **Attack 3** | No Detection | 8 | Complete evasion |'''
            
            new_table = '''| Attack | Objective | Pixels Used | Iterations | Expected Effectiveness |
|--------|-----------|-------------|------------|------------------------|
| **Attack 1** | Confidence Reduction | 30 | 500 | Moderate confidence reduction |
| **Attack 2** | Targeted Class | 30 | 500 | May produce target class detections |
| **Attack 3** | No Detection | 40 | 800 | High chance of complete evasion |

**Note**: Black-box pixel attacks are query-intensive. The strengthened parameters 
(30-40 pixels, 500-800 iterations) provide much better chances of success than 
the original conservative settings (5-8 pixels, 150-250 iterations).'''
            
            new_source = source.replace(old_table, new_table)
            if new_source != source:
                cell['source'] = new_source.split('\n')
                print("✓ Fixed summary table")

# Fix 6: Fix the misleading "success rate" calculation in final statistics
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'get_result_stats' in source and 'success_rate' in source:
            # Add a function to calculate proper success metrics
            if 'def get_attack_success_metrics' not in source:
                # Add the improved success metrics function
                improved_code = '''def get_result_stats(result: AttackResult, name: str, attack_type: str = "general") -> dict:
    """Extract statistics from attack result with proper success metrics."""
    orig_boxes = result.original_results.boxes
    adv_boxes = result.adversarial_results.boxes
    
    orig_n = len(orig_boxes) if orig_boxes else 0
    adv_n = len(adv_boxes) if adv_boxes else 0
    
    orig_conf = orig_boxes.conf.cpu().numpy().mean() if orig_boxes and len(orig_boxes) > 0 else 0
    adv_conf = adv_boxes.conf.cpu().numpy().mean() if adv_boxes and len(adv_boxes) > 0 else 0
    
    # Calculate success based on attack type
    success = False
    if attack_type == "no_detection":
        success = (adv_n == 0)  # Perfect: no detections
    elif attack_type == "confidence_reduction":
        success = (adv_conf < orig_conf * 0.5) and (adv_n <= orig_n)  # 50% confidence drop
    elif attack_type == "targeted_class":
        # Check if any detection has target class
        if adv_boxes and len(adv_boxes) > 0:
            classes = adv_boxes.cls.cpu().numpy().astype(int)
            success = any(c == TARGET_CLASS for c in classes)
    
    return {
        "name": name,
        "pixels": len(result.perturbations),
        "orig_detections": orig_n,
        "adv_detections": adv_n,
        "orig_conf": orig_conf,
        "adv_conf": adv_conf,
        "conf_reduction": orig_conf - adv_conf if orig_conf > 0 else 0,
        "detection_reduction": orig_n - adv_n,
        "success": success,
        "success_rate": 100.0 if success else 0.0,
        "best_fitness": result.best_fitness
    }

stats = [
    get_result_stats(attack1_result, "Confidence Reduction", "confidence_reduction"),
    get_result_stats(attack2_result, f"Targeted Class ({CLASS_NAMES[TARGET_CLASS]})", "targeted_class"),
    get_result_stats(attack3_result, "No Detection", "no_detection")
]'''
                
                # Find and replace the old function
                pattern = r'def get_result_stats\(result: AttackResult, name: str\) -> dict:.*?stats = \[.*?\]'
                if re.search(pattern, source, re.DOTALL):
                    new_source = re.sub(pattern, improved_code, source, flags=re.DOTALL)
                    cell['source'] = new_source.split('\n')
                    print("✓ Fixed success metrics calculation")

# Write the updated notebook
with open('black_box_pixel_attacks.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("\n" + "="*60)
print("All fixes applied successfully!")
print("="*60)
print("\nSummary of changes:")
print("1. Attack 1: k_pixels=30, n_iterations=500, population_size=20")
print("2. Attack 2: k_pixels=30, n_iterations=500, population_size=20")
print("3. Attack 3: k_pixels=40, n_iterations=800, population_size=25")
print("4. Fixed misleading analysis descriptions for all attacks")
print("5. Updated summary table with accurate expectations")
print("6. Added proper success metrics based on attack objectives")
