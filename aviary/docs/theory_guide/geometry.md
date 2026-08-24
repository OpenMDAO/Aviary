# Core Geometry Subsystem

The core geometry subsystem is responsible for calculating the geometric properties of the aircraft.
This includes the wing area, wing span, fuselage length, tail properties, passenger and cargo capacity, and other geometric properties.

```{note}
This section is under further development.
```

## Sizing the tails from volume coefficients

A tail volume coefficient relates the size of a tail to the wing it has to control.
It is the non-dimensional ratio

$$
  V = \frac{S_{tail} L_{tail}}{S_{wing} L_{ref}}
$$

where $S_{tail}$ is the tail area, $L_{tail}$ is the distance from the wing to the tail, and $S_{wing}$ is the wing area.
The reference length $L_{ref}$ is the wing mean aerodynamic chord for a horizontal tail, and the wing span for a vertical tail.

Rearranging that expression gives the tail area, which is what Aviary computes:

$$
  S_{tail} = \frac{V S_{wing} L_{ref}}{L_{tail}}
$$

GASP-based geometry always sizes both tails this way.
FLOPS-based geometry instead takes both tail areas directly from the user, because FLOPS itself treats them as inputs.
That works well when you model an existing aircraft, but it holds the tails at a fixed size while the rest of the aircraft changes.
If you resize the wing or optimize a new configuration, the tails no longer match the aircraft they belong to.

To size the tails in a FLOPS-based model, set either of these options:

| Option | Effect |
| --- | --- |
| `Aircraft.Design.COMPUTE_HTAIL_AREA` | Compute `Aircraft.HorizontalTail.AREA` instead of reading it |
| `Aircraft.Design.COMPUTE_VTAIL_AREA` | Compute `Aircraft.VerticalTail.AREA` instead of reading it |

Both options default to `False`, so existing models keep taking tail areas as inputs.

When you turn an option on, supply the volume coefficient and the moment arm for that tail:

| Variable | Meaning |
| --- | --- |
| `Aircraft.HorizontalTail.VOLUME_COEFFICIENT` | Horizontal tail volume coefficient |
| `Aircraft.HorizontalTail.MOMENT_ARM` | Distance from the wing to the horizontal tail |
| `Aircraft.VerticalTail.VOLUME_COEFFICIENT` | Vertical tail volume coefficient |
| `Aircraft.VerticalTail.MOMENT_ARM` | Distance from the wing to the vertical tail |

Both moment arms default to zero, which is not a usable value.
Aviary raises an error naming the variable if you enable an option without setting the matching moment arm.

The horizontal tail needs the wing mean aerodynamic chord.
FLOPS-based geometry does not provide `Aircraft.Wing.AVERAGE_CHORD`, so the component computes the mean aerodynamic chord of the trapezoidal wing from `Aircraft.Wing.AREA`, `Aircraft.Wing.SPAN`, and `Aircraft.Wing.TAPER_RATIO`.
Note that this differs from the mean geometric chord, $S_{wing} / b$, which FLOPS reports as `Aircraft.Wing.CHARACTERISTIC_LENGTH` and uses for skin friction.

Sizing a tail makes its area an output of the geometry subsystem.
If you also supply that area in your input file, Aviary's standard override behavior applies and your value takes precedence.
