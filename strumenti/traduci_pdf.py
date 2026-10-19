"""Translate the presentation while retaining the source PDF artwork."""
from pathlib import Path
import html
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[1]
T = {
1: ['Project 1: 3D Perception', 'Alessandro La Mantia'],
2: ['1. Title and Project Overview', '3D Perception with PointNet: From ModelNet40 Classification to Fine-Tuning on Horses', 'Environment and Tools: VS Code, Python, PyTorch, Streamlit, Plotly.', 'Key Objectives:', 'Understand and process unstructured 3D data (Point Clouds) using the PointNet neural network architecture.', 'Develop an interactive interface for exploring and inspecting the internal mechanisms of the network.', 'Conduct a fine-tuning experiment on a new unseen class (horse) and critically analyze the trade-off with existing classes.'],
3: ['2. The Learning Process', 'Work Methodology and Incremental Approach', 'Phase 1 — Theoretical Study: Understanding the challenges of 3D geometry (permutation invariance, rotation invariance) and the PointNet architecture.', 'Phase 2 — Inference Pipeline and Inspection: Building Parquet data reading, preprocessing, and internal feature visualization.', 'Phase 3 — Interactive App Development: Creating the Streamlit/Plotly dashboard to make theoretical concepts visible.', 'Phase 4 — Critical Experimentation: Designing the adaptation experiment for a new class (horses) and quantitatively evaluating the accuracy trade-off.'],
4: ['3. 3D Data: Point Clouds and the ModelNet40 Dataset', 'Direct Geometric Representation', 'Point Cloud: An unordered set of 3D points defined by (X, Y, Z) coordinates. Preserves continuous geometry without the complexity of 3D grids (Voxels) or 2D projections.', 'ModelNet40 Dataset: The ModelNet40 dataset contains 12,308 CAD objects, divided into 40 categories:', '9,840 training objects.', '2,468 test objects.', 'Main Challenge: The order of points in the file is arbitrary; the network must produce the same result regardless of how the points are ordered.'],
5: ['4. Essential Geometric Preprocessing', 'Sampling, Centering, and Normalization', 'Uniform Sampling: Reduction to 1,024 points per sample to ensure uniform input and computational efficiency.', "Spatial Normalization: Moving the center of the shape to the origin and scaling it to fit within a unit-radius sphere."],
6: ['5. PointNet Architecture: Key Concepts', 'How PointNet Handles Point Order and Transformation', '1. T-Net Modules (Spatial Transformers)', 'Learned transformation: computes 3×3 and 64×64 matrices to transform coordinates and features.', 'Adaptation: can rotate, stretch, or compress the point cloud differently for each object.', 'Objective: obtain a more effective representation for classification.', '2. Feature Extraction', 'Point-by-point processing: PointNet applies the same operations to each point independently.', 'Learned features: each point is represented by 1,024 numerical features.', '3. Global Max Pooling', 'For each feature, the maximum value across all 1,024 points is selected.', 'The result is a single global vector of 1,024 values representing the entire object.'],
7: ['6. PointNet Architecture: Essential Structure', 'From a Single Point to the Global Descriptor'],
8: ['7. Internal Inspection with PyTorch Forward Hooks', 'From Max Pooling to Classification', '1. Which points contribute to the descriptor?\nForward Hooks allow us to capture the activations of the 1,024 channels before Global Max Pooling. For each channel, we can identify the point that produced the maximum value.', '2. What do the colors indicate?\nThe color represents how many channels selected that point as their maximum. A high value indicates that the point contributes to the maximum of more features, but does not represent an importance percentage or automatically identify a specific part of the object.', '3. From descriptor to classification\nGlobal Max Pooling produces a global descriptor of 1,024 values. The final layers process this vector and produce 40 scores, one for each class. The class with the highest score determines the final prediction.'],
9: ['8. Main Interactive Application', '3D Exploration and Visualization (Streamlit & Plotly)', 'User Interface (localhost:8501):', 'Real-time 3D rendering of point clouds that can be manipulated and rotated.', 'Visualization of classification probabilities (Softmax) for the Top-5 categories.', 'Interactive inspection of Max Pooling activations with color mapping overlaid on the 3D model.'],
10: ['9. Difficulties and Solutions', 'The Main Technical Challenges of the Project', '💾 Data management\nVery large 3D dataset → read data only when needed, avoiding loading everything into RAM.', '🎲 Random sampling\nDifferent samples could change the prediction → use a fixed seed to obtain reproducible results.', '🔄 Point order invariance\nVerify that reordering points does not change the prediction → automated tests on the model.'],
11: ['Fine Tuning'],
12: ['11. Domain Extension: The "Horse" Class', 'Motivation and Research Question', 'The Context: The original ModelNet40 dataset does not include the "horse" category.', 'Strategic Research Question:', '"How many samples of the new class are needed to teach it to the model, and what is the cost in terms of performance degradation on the original 40 classes?"', 'Conceptual Approach: Evaluate Continual Learning capabilities and the prevention of Catastrophic Forgetting.'],
13: ['12. Dataset Preparation and Replay Strategy', 'Extraction from Objaverse and Forgetting Prevention', 'New Class Source: 3D horse models extracted from Objaverse (LVIS dataset) and sampled at 2,048 points.', 'Rigorous Deduplication: Checking and removing duplicates through geometric mesh hashing.', 'Replay Technique (Memory Buffer):', 'To prevent the network from "forgetting" the 40 original classes, 5 samples for each old class were included in the fine-tuning training set (200 replay samples in total).', 'Fixed Evaluation Test Set: 420 objects across classes (400 objects from the old classes + 20 unseen horses).'],
14: ['13. Fine-Tuning Strategy: Head Adaptation', 'Freezing the Feature Backbone', 'Backbone Freezing: The weights of the feature extraction modules (T-Net, Convolutions, Max Pooling) are frozen. The fundamental geometric representations remain unchanged.', 'Modifying the Last Layer:', 'The final classifier goes from 40 to 41 outputs.', 'The first 40 weights are initialized from the pre-trained weights; the 41st neuron is initialized for the horse class.', 'Computational Advantage: Extremely fast training, executable in a few seconds even on a CPU.'],
15: ['14. Fine-Tuning Implementation', 'Head Adaptation with PyTorch'],
16: ['15. Experimental Results: The Trade-Off', 'Quantitative Analysis of the Metrics', 'With 1–5 Horses: The network struggles to generalize. Horse Recall is low (< 40%).', 'With 10–25 Horses: An excellent balance is achieved: Horse Recall around 80–85% and minimal impact on the old classes.', 'With 50 Horses: Horse Recall rises to 95%, but Precision decreases and accuracy on the original 40 classes drops.', 'Deduction: Increasing the data for the new class without increasing replay samples leads the network to develop a bias toward the horse class, mistaking complex objects from old classes for horses.'],
17: ['16. Second App: Experiment Dashboard', 'Comparative Analysis of Results (Streamlit)', 'Experiment Interface (localhost:8502):', 'Interactive charts showing Recall, Precision, and Accuracy as the data budget varies.', 'Visual inspection of correctly or incorrectly classified test horses.', 'Direct comparison of predictions from models trained with 1 vs 50 horses.'],
18: ['17. Critical Analysis: Failures and Lessons Learned', 'What Did Not Work and How It Was Resolved', 'Initial Loss Imbalance:\nIn the initial trials without balancing the Loss weights, the network completely ignored the single horse presented (Recall = 0%) because the 40 classes dominated the gradient. This was resolved by introducing Weighted Cross-Entropy Loss.', 'The Illusion of High Recall:\nMeasuring only Recall for the new class gave the illusion that 50 horses were the perfect choice. Precision analysis revealed degradation on the old classes, highlighting the importance of balanced metrics.'],
19: ['18. Evaluation of Model and Dataset Limitations', 'Domain Shift and Generality Considerations', 'Domain Shift Between Datasets: Horses come from Objaverse (modern mesh models), while the 40 classes come from ModelNet40 (clean synthetic CAD). The difference in point density and distribution creates a "domain gap".', 'Noise in Real Data: Real point clouds (e.g., from LiDAR sensors) contain occlusions and noise that are absent from the clean synthetic data tested.', 'PointNet Limitations: PointNet processes each point individually before pooling; it does not capture detailed local geometric structures like more recent architectures (e.g., PointNet++ or DGCNN).'],
20: ['19. What Was Accomplished', 'Concrete Project Results', 'Point Cloud Pipeline: loading data from Parquet, sampling 1,024 points, centering and normalizing the point cloud.', 'PointNet Analysis: study of T-Net, feature extraction, and Global Max Pooling, up to the global vector of 1,024 features.', 'Internal Inspection: using Forward Hooks to identify which points contribute to feature maxima and visualize them on the point cloud.', 'Fine-Tuning: adding the new horse class by going from 40 to 41 classes, while retaining replay of 200 samples from the original classes.', 'Evaluation: comparing performance on the same test set of 420 objects to verify recognition of the new class and retention of previously learned classes.'],
21: ['20. Conclusions and Future Developments', 'Future Developments', 'Test Larger Amounts of Data: repeat fine-tuning with more examples of the horse class to study how the model’s learning capacity changes.', 'Improve Fine-Tuning: compare the current Head Adaptation with partial fine-tuning of the network, progressively unfreezing some backbone layers.', 'Analyze Features More Thoroughly: extend the activation visualization to understand which parts of the point cloud most influence classification.', 'Expand the Evaluation: repeat the experiments with more classes and more objects to verify whether the results observed for horses generalize to other cases.'],
22: ['Thank you for your attention'],
}

COMMENTS = {
'Generatore casuale con seed fisso: sempre lo stesso campionamento': 'Random generator with a fixed seed: always the same sampling',
'Seleziona casualmente num_punti punti dalla point cloud senza ripetizioni': 'Randomly select num_punti points from the point cloud without repetition',
"Sposta la point cloud in modo che il suo centro coincida con l'origine": 'Shift the point cloud so that its center coincides with the origin',
"Calcola la distanza massima dei punti dall'origine": 'Compute the maximum distance of the points from the origin',
'Normalizza': 'Normalize',
'Struttura concettuale del passaggio Forward in PointNet': 'Conceptual structure of the Forward pass in PointNet',
'x ha forma:': 'x has shape:', 'T-Net spaziale': 'Spatial T-Net',
'Allineamento geometrico': 'Geometric alignment',
'Estrazione feature locali': 'Local feature extraction',
'T-Net feature': 'Feature T-Net',
'Mappatura ad alta dimensione': 'High-dimensional mapping',
'Invarianza permutazione': 'Permutation invariance',
'Classificazione finale (40 classi)': 'Final classification (40 classes)',
'Nuovo classificatore: 40 categorie originali + cavallo.': 'New classifier: 40 original categories + horse.',
'Copiamo i pesi delle categorie già conosciute.': 'Copy the weights of the already known categories.',
'Metà del peso ai cavalli, metà alle vecchie categorie.': 'Half the weight to horses, half to the old categories.',
}

def main():
    doc = pdf.open(ROOT / 'Presentazione Progetto 1.pdf')
    scales = []
    for number, page in enumerate(doc, 1):
        blocks = [b for b in page.get_text('dict')['blocks'] if 'lines' in b]
        translations = iter(T[number])
        jobs = []
        for block in blocks:
            lines = block['lines']
            text = '\n'.join(''.join(s['text'] for s in l['spans']) for l in lines)
            spans = [s for l in lines for s in l['spans']]
            code = (number == 5 and text.startswith('def ')) or (number == 7 and text.startswith('# ')) or (number == 15 and block['bbox'][1] > 100 and block['bbox'][1] < 520)
            footer = text.startswith('Progetto 3D Perception')
            if code:
                translated = text
                for old, new in COMMENTS.items():
                    translated = translated.replace(old, new)
            elif footer:
                translated = text.replace('Progetto 3D Perception', '3D Perception Project')
            else:
                translated = next(translations)
            # Keep already-English content and code segments byte-for-byte in the PDF.
            if translated == text:
                continue
            rect = pdf.Rect(block['bbox'])
            if footer:
                # The original footer is a single extraction block with two distant labels.
                rect.x1 = 250
                translated = '3D Perception Project'
                redactions = [pdf.Rect(l['bbox']) for l in lines[:1]]
            else:
                redactions = [pdf.Rect(s['bbox']) for s in spans]
            # Preserve the three original emoji glyphs.
            if number == 10 and rect.y0 > 100 and not footer:
                first = spans[0]
                redactions = [pdf.Rect(s['bbox']) for s in spans[1:]] if len(first['text']) == 1 else redactions
                if len(first['text']) == 1:
                    rect.x0 = spans[1]['bbox'][0]
                    translated = translated[2:] if translated[1:2] == ' ' else translated
            for r in redactions:
                page.add_redact_annot(r, fill=False)
            size = spans[0]['size']
            color = spans[0]['color']
            jobs.append((rect, translated, size, color, code, footer, len(lines)))
        try:
            next(translations)
            raise ValueError(f'Unused translations on page {number}')
        except StopIteration:
            pass
        page.apply_redactions(images=0, graphics=0)
        for rect, text, size, color, code, footer, line_count in jobs:
            if code:
                # Exact original line positions, indentation, and executable identifiers.
                for offset, line in enumerate(text.split('\n')):
                    if not line.strip():
                        continue
                    y = rect.y0 + offset * ((rect.height - size * 1.15) / max(line_count - 1, 1))
                    r = pdf.Rect(rect.x0, y, 796, y + size * 1.6)
                    markup = '<div style="white-space:pre">' + html.escape(line).replace(' ', '&#160;') + '</div>'
                    page.insert_htmlbox(r, markup, css=f'*{{margin:0;padding:0}} div{{font-family:monospace;font-size:{size}pt;line-height:1;color:#{color:06x}}}')
                continue
            centered = number in (1, 22)
            if centered:
                rect.x0, rect.x1 = 45.4, 796.5
            elif not footer:
                rect.x1 = 796.5
            rect.y1 += 3
            weight = 'bold' if size > 20 or (len(text.splitlines()) == 1 and text.endswith(':')) else 'normal'
            chunks = text.split('\n')
            markup = '<div>' + '<br>'.join(html.escape(s) for s in chunks) + '</div>'
            align = 'center' if centered else 'left'
            css = f'*{{margin:0;padding:0}} div{{font-family:Arial,sans-serif;font-size:{size}pt;line-height:1.4;color:#{color:06x};font-weight:{weight};text-align:{align}}}'
            spare, scale = page.insert_htmlbox(rect, markup, css=css)
            assert spare >= 0, (number, text)
            scales.append((number, round(scale, 3)))
    output = ROOT / 'Presentazione Progetto 1 - English.pdf'
    doc.save(output, garbage=4, deflate=True)
    check = pdf.open(output)
    assert len(check) == 22
    for number, page in enumerate(check, 1):
        assert page.get_text().strip(), number
    print(output.name, '—', len(check), 'pages')
    print('Minimum text scale:', min(s[1] for s in scales))
    # Render contact sheets for visual verification.
    for start in (0, 11):
        preview = pdf.open()
        sheet = preview.new_page(width=1263, height=1272)
        for idx in range(start, min(start + 11, len(check))):
            n = idx - start
            x, y = (n % 3)*421, (n // 3)*318
            sheet.show_pdf_page(pdf.Rect(x, y, x+421, y+297.5), check, idx)
            sheet.insert_text((x+4, y+311), str(idx+1))
        sheet.get_pixmap().save(ROOT / f'.translation-preview-{start}.png')

if __name__ == '__main__':
    main()
