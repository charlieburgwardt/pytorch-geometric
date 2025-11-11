
# Solving the Line-of-Sight (LOS) Equation for Vehicle Speed

This document provides a step-by-step derivation for solving the Line-of-Sight (LOS) equation to isolate the vehicle speed $ v $, assuming the vehicle heading and all other parameters are known.

## General LOS Equation

The LOS equation relates the relative motion between a vehicle and a target in terms of their positions and velocities. The general form of the LOS rate equation is:

$$
\dot{\theta} = \frac{v \sin(\psi - \theta)}{R}
$$

Where:

- $ \dot{\theta} $: LOS rate (rate of change of LOS angle)
- $ v $: Vehicle speed (unknown)
- $ \psi $: Vehicle heading angle (known)
- $ \theta $: LOS angle (known)
- $ R $: Range between the vehicle and the target (known)

## Objective

Solve the LOS equation for the vehicle speed $ v $.

## Step-by-Step Derivation

Starting with the LOS equation:

$$
\dot{\theta} = \frac{v \sin(\psi - \theta)}{R}
$$

Multiply both sides by $ R $:

$$
R \dot{\theta} = v \sin(\psi - \theta)
$$

Now isolate $ v $:

$$
v = \frac{R \dot{\theta}}{\sin(\psi - \theta)}
$$

## Final Expression

The vehicle speed $ v $ is given by:

$$
v = \frac{R \dot{\theta}}{\sin(\psi - \theta)}
$$

This expression allows you to compute the vehicle speed when the LOS rate, vehicle heading, LOS angle, and range are known.

## Notes

- Ensure that $ \sin(\psi - \theta) \neq 0 $ to avoid division by zero.
- Angles should be in radians when performing trigonometric calculations.

