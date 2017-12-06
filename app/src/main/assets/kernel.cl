__kernel void simpleMultiply(
    __global uchar* dest_data,
    __global uchar* src_data,
    int width,
    int height,
    int channel,
    float cosThreta,
    float sinThreta)
{
  const int ix = get_global_id(0);
  const int iy = get_global_id(1);

  int xpos = ((float)(ix - width / 2)) * cosThreta + ((float)(-iy + height / 2)) * sinThreta + width / 2;
  int ypos = ((float)(ix - width / 2)) * sinThreta + ((float)(iy - height / 2)) * cosThreta + height / 2;

  if((xpos >= 0) && (xpos < width) && (ypos >= 0) && (ypos <= height))
  {
    dest_data[ypos * width + xpos] = src_data[iy * width + ix];
  }
}