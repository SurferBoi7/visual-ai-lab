import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"

import { cn } from "@/lib/utils"

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center group",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track className="relative h-px w-full grow overflow-hidden rounded-full bg-white/10">
      <SliderPrimitive.Range className="absolute h-full bg-[#0A84FF]" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb className="block h-3 w-3 rounded-full bg-white shadow-[0_0_0_3px_rgba(10,132,255,0.25)] transition-shadow focus-visible:outline-none focus-visible:shadow-[0_0_0_4px_rgba(10,132,255,0.4)] hover:shadow-[0_0_0_4px_rgba(10,132,255,0.35)] disabled:pointer-events-none disabled:opacity-50" />
  </SliderPrimitive.Root>
))
Slider.displayName = SliderPrimitive.Root.displayName

export { Slider }
