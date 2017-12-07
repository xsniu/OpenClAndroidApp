const sampler_t mysampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void simpleMultiply(
    __write_only image2d_t dest_data,
    __read_only image2d_t src_data,
    int cols,
    int rows,
   __constant float *filter,
   int filterSize)
{
  int col = get_global_id(0);
  int row = get_global_id(1);
  int halfWidth = (int)(filterSize / 2);

  uint4 sum = {0, 0, 0, 0};
  int filterIdx = 0;

  int2 coords;

  for (int i = -halfWidth; i <= halfWidth; i++)
  {
    coords.y = row + i;
    for (int j = -halfWidth; i <= halfWidth; i++)
    {
      coords.x = col + j;
      uint4 pixel;
      pixel = read_imageui(src_data, mysampler, coords);
      sum.x += ((float)pixel.x) * filter[filterIdx++];
    }
  }


  //uint4 res;
  //res.x = sum.x;
  if ((row < rows) && (col < cols))
  {
    coords.x = col;
    coords.y = row;
    write_imageui(dest_data, coords, sum);
  }
}