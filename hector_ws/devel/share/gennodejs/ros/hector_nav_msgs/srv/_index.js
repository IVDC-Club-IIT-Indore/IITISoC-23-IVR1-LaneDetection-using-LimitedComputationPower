
"use strict";

let GetRobotTrajectory = require('./GetRobotTrajectory.js')
let GetNormal = require('./GetNormal.js')
let GetRecoveryInfo = require('./GetRecoveryInfo.js')
let GetSearchPosition = require('./GetSearchPosition.js')
let GetDistanceToObstacle = require('./GetDistanceToObstacle.js')

module.exports = {
  GetRobotTrajectory: GetRobotTrajectory,
  GetNormal: GetNormal,
  GetRecoveryInfo: GetRecoveryInfo,
  GetSearchPosition: GetSearchPosition,
  GetDistanceToObstacle: GetDistanceToObstacle,
};
