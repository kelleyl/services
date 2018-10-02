import bars_and_tones
import bottom_thirds
import generate_text_boxes

video_path = "../videos/cpb-aacip-507-0c4sj1b49m.mp4"
s0 = generate_text_boxes.Text_Boxes(video_path)
print (s0.run_service())
s1 = bars_and_tones.Bars_Tones(video_path)
print (s1.run_service())
s2 = bottom_thirds.Bottom_Thirds(video_path)
print (s2.run_service())

