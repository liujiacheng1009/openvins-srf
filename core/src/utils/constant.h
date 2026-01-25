#pragma once

constexpr double kRad2Deg = 180.0 / M_PI;
constexpr double kDeg2Rad = M_PI / 180.0;
constexpr double kGravityMag = 9.81;

// time unit conversion
constexpr double kSecToMsec = 1000UL;
constexpr double kSecToUsec = 1000'000UL;
constexpr double kSecToNsec = 1000'000'000UL;

constexpr double kMsecToSec = 1.e-3;
constexpr double kMsecToUsec = 1000UL;
constexpr double kMsecToNsec = 1000'000UL;

constexpr double kUsecToSec = 1.e-6;
constexpr double kUsecToMsec = 1.e-3;
constexpr double kUsecToNsec = 1000UL;

constexpr double kNsecToSec = 1.e-9;
constexpr double kNsecToMsec = 1.e-6;
constexpr double kNsecToUsec = 1.e-3;

constexpr double kEpsilon = 1.e-9;

//! State Local Dimension
namespace LDim
{
    constexpr int Landmark = 3;
    constexpr int TimeOffset = 1;
    constexpr int Ori = 3;
    constexpr int Tvec = 3;
    constexpr int Vel = 3;
    constexpr int Bias = 6;
    constexpr int BiasGyro = 3;
    constexpr int BiasAccel = 3;
    constexpr int Pose = 6;
    constexpr int PoseVel = 9;
    constexpr int PoseVelBias = 15;
}