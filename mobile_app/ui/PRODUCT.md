# Product


## Platform

android

## Users
A driver (car, truck, or two-wheeler rider) in India using their smartphone mounted on the dashboard as their main navigation device. They are driving through GPS-denied environments (tunnels, underground parking, dense urban areas, forested highways) and need to glance at the screen quickly without losing focus on the road.

## Product Purpose
To provide uninterrupted, highly accurate vehicle positioning when GPS signals weaken or disappear entirely. It serves as a dashboard instrument that proves the viability of AI-corrected motion sensor data (accelerometer + gyroscope) for continuous navigation. Success is defined by presenting this capability as technically credible to hackathon judges (ISRO) while remaining glanceable for drivers.

## Positioning
Unlike standard navigation apps (Google Maps) which freeze, jump, or show wrong positions when GPS is lost, this app uses an AI-corrected INS (Inertial Navigation System) for seamless dead reckoning, making the tracking mode transparent to the user so they always know the reliability of their position estimate.

## Operating Context
- Phone mounted on a vehicle dashboard
- Driver is actively driving
- Rapid glances only; visual hierarchy must prioritize extreme legibility (e.g., massive speed indicator) over dense reading
- Transitions between clear skies (GPS active) and denied environments (tunnels, urban canyons)

## Capabilities and Constraints
- Must function primarily via sensor fusion when GPS is lost
- Must clearly show two distinct, unambiguous states: "GNSS + INS Fusion" vs "Dead Reckoning"
- Prototype for ISRO (Smart India Hackathon); requires high technical credibility, not just consumer-friendly aesthetics

## Brand Commitments
- Visual style: Premium automotive/aviation instrumentation
- Colors: Restrained palette. Cyan/teal for normal operations, amber/orange for warning/dead-reckoning mode
- Typography: Precise, high contrast, clear hierarchy
- Bans: No purple-to-blue gradients, no emoji, no generic centered-everything layouts, no rounded-icon-tile-above-heading pattern

## Product Principles
1. **Instrument, not an app:** Design for a driver's glance, not a user's scroll. Prioritize massive key metrics over supporting text.
2. **Absolute transparency:** Never hide the system's state; the transition between GNSS and Dead Reckoning must be the most visually unambiguous element.
3. **Technical credibility:** Look like professional aerospace/automotive hardware built for ISRO, avoiding playful or generic consumer tropes.
