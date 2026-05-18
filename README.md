## Methodik: Hintergrund

### Vision Transformer (ViT)
Im Gegensatz zu herkömmlichen Modellen (CNNs), die Bilder lokal Pixel für Pixel scannen, zerlegt ein Transformer das Bild in kleine Quadrate (**Patches**).
* **Globaler Kontext:** Das Modell lernt die Beziehung jedes Patches zu allen anderen im gesamten Bild gleichzeitig.
* **Attention-Mechanismus:** Der Transformer entscheidet aktiv, welche Bereiche (z. B. Rost-Texturen) für die Analyse relevant sind und welche (z. B. der Hintergrund) ignoriert werden können. Dies ermöglicht eine präzisere Unterscheidung zwischen Bauteil und Umgebung.

### DINOv2 (Self-Supervised Learning)
DINOv2 ist ein von Meta AI entwickeltes Trainingsverfahren, das ohne manuelle Beschriftungen (Labels) auskommt.
* **Feature-Extraktion:** Das Modell lernt die "Sprache" von Oberflächen und Materialien rein durch den automatisierten Vergleich verschiedener Bildausschnitte.
* **Textur-Verständnis:** Durch gezieltes Fine-Tuning auf Rost-Datensätzen erkennt DINOv2 die spezifische visuelle Signatur von Korrosion als Materialeigenschaft, unabhängig von der Form des Bauteils (Auto, Brücke oder Stahlträger).
* **Register-Tokens:** Wir verwenden eine Architektur mit sogenannten "Registern". Diese verhindern "Halluzinationen" (falsche Aktivierungen) in homogenen Bildbereichen wie Wänden und halten die Aufmerksamkeit stabil auf dem metallischen Werkstück.