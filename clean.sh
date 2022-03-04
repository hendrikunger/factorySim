#!/bin/bash

#rm -R models
rm -R Output

#mkdir models
mkdir Output

touch ./Output/.gitkeep

echo "#!/bin/sh" > ./Output/slideshow.sh
echo "#ffmpeg -y -framerate 4/1  -pattern_type glob -i '*.png' -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4" >> ./Output/slideshow.sh
echo "ffmpeg -y -f image2 -framerate 4/1 -pattern_type glob -i '*.png' -vf scale=500x500 out.gif" >> ./Output/slideshow.sh
echo "find -type f -iname '*.png' -delete" >> ./Output/slideshow.sh

chmod +x ./Output/slideshow.sh


