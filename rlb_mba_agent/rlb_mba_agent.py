

##################################################################################################################
"""
"""

# Built-in/Generic Imports
from distutils.debug import DEBUG
import json
import os
import queue
import math
import random
from datetime import datetime
from functools import partial
import json
import time

import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import JointState, LaserScan
from rlb_utils.msg import Goal, TeamComm, CommsState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import pandas as pd
import math

from rlb_tools.Caylus_map_loader import load_maps
from rlb_tools.Raster_ray_tracing import check_comms_available
from rlb_config.simulation_parameters import *

##################################################################################################################


DEBUG = False

class RLB_mba_agent(Node):
    def __init__(self):
        # -> Initialise inherited classes
        Node.__init__(self, 'mba_agent')

        # -> Setup classes
        self.verbose = 0
        self.msg_buffer = []

        # -> Setup robot ID
        self.declare_parameter('robot_id', 'Turtle')
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value

        # ----------------------------------- Filter output subscriber
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            )

        self.filter_subscriber = self.create_subscription(
            msg_type=TeamComm,
            topic=f"/{self.robot_id}/sim/filter_output",
            callback=self.process_msg,
            qos_profile=qos
        )

        # ----------------------------------- Team comms publisher
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            )

        self.team_comms_publisher = self.create_publisher(
            msg_type=TeamComm,
            topic="/team_comms",
            qos_profile=qos 
        )


        # ================================================================= MBA agent
        self.tasks = {}
        self.tokens = {}
        self.memory = {}

        #-> Setup processing loop
        time.sleep(5)
        self.processor = self.create_timer(
            timer_period_sec=.1,
            callback=self.run_auction
        )

    def process_msg(self, msg):
        if msg.type == "auction_token":
            token = json.loads(msg.data)
            # self.get_logger().info(f"Received token: {token}")
            self.check_auction_token(token=token)

    @property
    def current_auction_timestamp(self):
        dt = datetime.now()
        timestamp = datetime.timestamp(dt)
        return int(timestamp)

    def run_auction(self):
        self.share_current_auction_token()
    
    def check_auction_token(self, token):
        session = self.current_auction_timestamp

        # -> If token is of current auction
        if token["session"] == session:
            # -> Create current auction token if it does not exist
            if session not in self.tokens.keys():
                self.tokens[session] = self.create_auction_token(session=session)

            # -> Do not compare if tokens have same holder_id
            if self.tokens[session]['holder_id'] == token['holder_id']:
                return

            # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Comparing token values ({self.tokens[session]['value']} vs {token['value']})")
            # -> If local token has a larger value, keep local token
            if self.tokens[session]["value"] > token["value"]:
                # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has higher value ({self.tokens[session]['value']} vs {token['value']})")
                pass

            # -> If new token has larger value, keep new token
            elif self.tokens[session]["value"] < token["value"]:
                if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has higher value ({self.tokens[session]['value']} vs {token['value']})")
                self.tokens[session] = token

            # -> If local and new tokens have the same value, compare priorities
            else:
                # if DEBUG: self.get_logger().warning(f"Session {str(session)[-2:]} - Tokens have same value, falling back to priority ({self.tokens[session]['priority'] } vs {token['priority']})")
                
                # -> If local token has a larger priority, keep local token
                if self.tokens[session]["priority"] > token["priority"]:
                    # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has higher priority ({self.tokens[session]['priority'] } vs {token['priority']})")
                    pass

                # -> If new token has larger priority, keep new token
                elif self.tokens[session]["priority"] < token["priority"]:
                    if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has higher priority ({self.tokens[session]['priority'] } vs {token['priority']})")
                    self.tokens[session] = token

                # -> If local and new tokens have the same priority, keep token with lowest holder_id
                else:
                    # if DEBUG: self.get_logger().warning(f"Session {str(session)[-2:]} - Tokens have same priority, falling back to id ({self.tokens[session]['holder_id']} vs {token['holder_id']})")
                    
                    if int(self.tokens[session]["holder_id"].split("_")[-1]) < int(token["holder_id"].split("_")[-1]):
                        # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has lower id ({self.tokens[session]['holder_id']} vs {token['holder_id']})")
                        pass

                    else:
                        if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has lower id ({self.tokens[session]['holder_id']} vs {token['holder_id']})")
                        self.tokens[session] = token

    def share_current_auction_token(self):
        session = self.current_auction_timestamp

        if session not in self.tokens.keys():
            self.tokens[session] = self.create_auction_token(session=session)

        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_token"
        msg.data = json.dumps(self.tokens[session])

        self.team_comms_publisher.publish(msg)

    def create_auction_token(self, session):
        # t_id = []
        # for token in self.tokens.values():
        #     t_id.append((str(token["session"])[-2:], token["holder_id"][-1]))
        # self.get_logger().info(str(t_id))

        return {
            "session": session,
            "holder_id": self.robot_id,
            "priority": 0,
            # "value": 1
            "value": random.randint(0, 10)
        }

def main(args=None):
    # `rclpy` library is initialized
    rclpy.init(args=args)

    path_sequence = RLB_mba_agent()

    rclpy.spin(path_sequence)

    path_sequence.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
