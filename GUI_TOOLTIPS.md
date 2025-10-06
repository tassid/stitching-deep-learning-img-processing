# Descrições dos Métodos para Tooltips

## Métodos Clássicos:

1. **OpenCV Stitcher** - Stitcher oficial do OpenCV com auto-detecção. Rápido e confiável.

2. **ORB + Homography** - Features ORB + RANSAC. Bom para imagens com pouca textura.

3. **Blend Poisson** - Seamless cloning de Poisson nas áreas de overlap. Ótimo para remover costuras.

4. **Blend Multifocal** - Blend baseado em foco. Combina áreas nítidas de múltiplas imagens.

5. **Blend Hybrid** - Combina contraste, saturação e exposição. Flexível e personalizável.

6. **ECC Alignment** - Enhanced Correlation Coefficient para alinhamento preciso.

7. **Stitcher Básico** - SIFT + homografia simples. Base sólida e rápida.

8. **Manual SIFT RANSAC Feather** - SIFT manual com feather blending suave.

9. **Features Homography** - Features clássicos (SIFT/ORB) com múltiplas opções.

10. **Blend Exposure** - Feather blend suave baseado em distância. Natural e rápido.

11. **Blend Wavelet** - Fusão por wavelets (Daubechies). Preserva detalhes finos.

12. **Seams Exposure Multiband** - Graph-cut seams + multiband blending. Resultado profissional.

13. **Detectors Harris Fast** - Detectores de cantos Harris/FAST. Leve e rápido.

14. **Blend Laplacian** - Pirâmide Laplaciana para blend multi-escala. Suave e natural.

15. **Exposure Comp Only** - Compensação de exposição + feather. Remove diferenças de iluminação.

## Métodos Deep Learning:

16. **Deep Homography** - CNN para estimativa de homografia. Robusto a grandes deslocamentos.

17. **Flow Stitch** - SIFT robusto com fallback. Confiável para panoramas gerais.

18. **LoFTR** - Detector-free matching. Excelente para pares com pouca textura.

19. **Sp Superglue** - SIFT com blend adaptativo. Robusto e versátil.

20. **UDIS** - SIFT simplificado. Rápido e eficiente.
