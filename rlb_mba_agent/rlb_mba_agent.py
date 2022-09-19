

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
from unittest import result
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import JointState, LaserScan
from rlb_utils.msg import Goal, TeamComm, CommsState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from patrol_messages.msg import BundleDescription, Award, Results, ResultsStamped, Header, AuctionHeader


import numpy as np
import pandas as pd
import math

from patrol_robot.robot_member_function import Robot

from rlb_tools.Caylus_map_loader import load_maps
from rlb_tools.Raster_ray_tracing import check_comms_available
from rlb_config.simulation_parameters import *

##################################################################################################################


DEBUG = True

class RLB_mba_agent(
    # Node, 
    Robot):
    def __init__(self):
        # -> Initialise inherited classes
        # Node.__init__(self, 'mba_agent')

        Robot.__init__(self)
        
        # -> Setup classes
        self.verbose = 0
        self.msg_buffer = []

        # -> Setup auction properties
        self.auction_cycle_length = 5

        self.consensus_phase_length = 1/3
        self.bidding_phase_length = 1/3
        self.awarding_phase_length = 1/3

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

        self.sim_clock_subscriber = self.create_subscriber(
            msg_type=TeamComm,
            topic="/RLB_clock",
            callback=self.sim_clock_callback,
            qos_profile=qos 
        )

        # ================================================================= MBA agent
        self.tasks = {}
        self.tokens = {}
        self.auction_bids = {}
        self.sim_time = 0

        #-> Setup processing loop
        time.sleep(5)
        self.processor = self.create_timer(
            timer_period_sec=.5,
            callback=self.run_auction_cycle
        )

    @property
    def current_timestamp(self):
        dt = datetime.now()
        return datetime.timestamp(dt)

    @property
    def current_simulation_time(self):
        return self.sim_time

    @property
    def current_session_id(self):
        current_timestamp = self.current_timestamp
        return current_timestamp - (current_timestamp % self.auction_cycle_length)

    @property
    def current_auction_phase(self):
        current_auction_progress = (self.current_timestamp % self.auction_cycle_length) / self.auction_cycle_length

        if 0 <= current_auction_progress < self.consensus_phase_length:
            return "consensus"

        elif  self.consensus_phase_length <= current_auction_progress < self.consensus_phase_length + self.bidding_phase_length:
            return "biding"

        else:
            return "awarding"

    # -------------------------------------------------- Auction process
    def sim_clock_callback(self, msg):
        self.sim_time = (msg.data)

    def process_msg(self, msg):       
        """
        Three checks are performed:
            - If the msg belongs to one of the handled msg types
            - If the msg type corresponds to the current auction phase
            - If the msg has already been broadcasted once by the agent (using trace)

        Msgs that fail any is not rebroadcased
        """ 

        # self.get_logger().info(f"====================================> self.current_auction_phase: {self.current_auction_phase}")
        if msg.type == "auction_token":
            if self.current_auction_phase == "consensus":
                self.check_auction_token(msg=msg)
            else:
                return

        elif msg.type == "auction_bid":
            if self.current_auction_phase == "biding":
                self.check_auction_bid(msg=msg)
            else:
                return

        elif msg.type == "auction_award":
            if self.current_auction_phase == "awarding":
                self.check_auction_award(msg=msg)
            else:
                return
        
        else:
            return
        
        # -> Check/Update msg trace
        msg, trace_flag = self.handle_msg_trace(msg=msg)

        # -> Re-broadcast message if not done before
        if trace_flag:
            self.team_comms_publisher.publish(msg)

    def run_auction_cycle(self):
        if self.current_auction_phase == "consensus":
            self.share_current_auction_token()

        elif self.current_auction_phase == "biding":
            if self.current_session_id in self.tokens.keys():
                self.get_logger().info(f"{self.robot_id}: {str(self.current_session_id)[-3:]} - {self.tokens[self.current_session_id]['value']}")
                self.share_current_auction_bid()
        
        else:
            if self.current_session_id in self.tokens.keys():
                if self.tokens[self.current_session_id]['holder_id'] == self.robot_id:
                    self.share_current_auction_award()

    # -------------------------------------------------- Auction award
    def check_auction_award(self, msg):
        session = self.current_session_id

        # -> If bid is of current auction
        if award["session"] == session:
            if award['holder_id'] == self.robot_id:
                award = json.loads(msg.data)
                
                # TODO: Assign task to robot with id self.robot_id
                # -> Construct Award msg
                award_msg = Award()
                award_msg.bidder_id = self.robot_id
                
                # -> Construct BundleDescription msg
                bundle_description_msg = BundleDescription()
                bundle_description_msg.bundle_id = self.tokens[session]["task_id"]
                bundle_description_msg.item_bundle = [self.tokens[session]["task_item"]]

                # -> Construct header msg
                header_msg = Header()
                header_msg.sender_id = self.robot_id
                
                # -> Construct AuctionHeader msg
                auction_header_msg = AuctionHeader()
                auction_header_msg.auction_id = str(session)

                # -> Construct Results msg
                results_msg = Results()
                results_msg.auction_header = auction_header_msg
                results_msg.award = [award_msg]
                results_msg.bundle_descriptions = [bundle_description_msg]
                
                # -> Construct ResultsStamped msg
                results_stamped_msg = ResultsStamped()
                results_stamped_msg.header = header_msg
                results_stamped_msg.body_msg = results_msg

                pass

    def share_current_auction_award(self):
        # -> Construct award msg
        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_award"
        msg.data = json.dumps(self.create_auction_award())
        msg.trace = [self.robot_id] 

        self.team_comms_publisher.publish(msg)

    def create_auction_award(self):
        session = self.current_session_id
        
        # Create bid msg list
        if not DEBUG:
            msg.bundle = [self.tokens[session]["task_item"]]

            bids = []
            best_bidder = self.auction_scheme.solve_wdp(bids)[0]
            
            # TODO: Run auctioneer self award

            return {
                "session": self.current_session_id,
                "holder_id": best_bidder
            }
        else:

            return {
                "session": self.current_session_id,
                "holder_id": self.robot_id
            }

    # -------------------------------------------------- Auction bid
    def check_auction_bid(self, msg):
        bid = json.loads(msg.data)
        session = self.current_session_id

        # -> If bid is of current auction
        if bid["session"] == session:
            # -> Check if participating in auction
            if session not in self.tokens.keys():
                return

            # -> If token holder for this auction
            if self.tokens[session]['holder_id'] == self.robot_id:
                # -> Check if bid list exists for existing session
                if session not in self.auction_bids.keys():
                    self.auction_bids[session] = []

                # -> Check if current auction entry exists in self.auction_bids
                if bid not in self.auction_bids[session]:
                    self.auction_bids[session].append(bid)

                else:
                    # Bid already registered, pass
                    pass

            else:
                # Not auction token holder, pass
                pass
    
    def share_current_auction_bid(self):
        # -> Construct bid msg
        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_bid"
        msg.data = json.dumps(self.create_auction_bid())
        msg.trace = [self.robot_id] 

        self.team_comms_publisher.publish(msg)

    def create_auction_bid(self):
        session = self.current_session_id

        if not DEBUG:
            task_value = self.auction_scheme.compute_bid(self.tokens[session]["task_item"])[0]

            return {
                "session": self.current_session_id,
                "task_item": self.tokens[session]["task_item"],
                "holder_id": self.robot_id,
                "priority": 0,
                "value": task_value
            }

        else:
            return {
                "session": self.current_session_id,
                "task_item": self.tokens[session]["task_item"],
                "holder_id": self.robot_id,
                "priority": 0,
                "value": random.randint(0, 10)
            }

    # -------------------------------------------------- Auction token consensus
    def check_auction_token(self, msg):
        token = json.loads(msg.data)
        session = self.current_session_id

        # -> If token is of current auction
        if token["session"] == session:
            # -> Create local token if does not exist
            if session not in self.tokens.keys():
                self.tokens[session] = self.create_auction_token(session=session)

            # -> Compare local token with token received, key better one
            self.tokens[session] = self.compare_tokens(token_1=self.tokens[session], token_2=token)

    def compare_tokens(self, token_1, token_2):
        session = self.current_session_id
        
        if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Comparing token \n{token_1} \nvs \n{token_2}")
        
        # -> If tokens are of comparable session
        if token_1["session"] == token_2["session"]:

            # -> Do not compare if tokens have same holder_id
            if token_1['holder_id'] == token_2['holder_id']:
                return token_1

            # Consensus
            # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Comparing token values ({token_1['value']} vs {token['value']})")
            # -> If local token has a larger value, keep local token
            if token_1["value"] > token_2["value"]:
                # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has higher value ({token_1['value']} vs {token['value']})")
                return token_1

            # -> If new token has larger value, keep new token
            elif token_1["value"] < token_2["value"]:
                if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has higher value ({token_1['value']} vs {token_2['value']})")
                return token_2

            # -> If local and new tokens have the same value, compare priorities
            else:
                # if DEBUG: self.get_logger().warning(f"Session {str(session)[-2:]} - Tokens have same value, falling back to priority ({token_1['priority'] } vs {token['priority']})")
                
                # -> If local token has a larger priority, keep local token
                if token_1["priority"] > token_2["priority"]:
                    # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has higher priority ({token_1['priority'] } vs {token['priority']})")
                    return token_1

                # -> If new token has larger priority, keep new token
                elif token_1["priority"] < token_2["priority"]:
                    if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has higher priority ({token_1['priority'] } vs {token_2['priority']})")
                    return token_2

                # -> If local and new tokens have the same priority, keep token with lowest holder_id
                else:
                    # if DEBUG: self.get_logger().warning(f"Session {str(session)[-2:]} - Tokens have same priority, falling back to id ({token_1['holder_id']} vs {token['holder_id']})")
                    
                    if int(token_1["holder_id"].split("_")[-1]) < int(token_2["holder_id"].split("_")[-1]):
                        # if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Current token has lower id ({token_1['holder_id']} vs {token['holder_id']})")
                        return token_1

                    else:
                        if DEBUG: self.get_logger().info(f"Session {str(session)[-2:]} - Received token has lower id ({token_1['holder_id']} vs {token_2['holder_id']})")
                        return token_2

    def share_current_auction_token(self):
        session = self.current_session_id

        if session not in self.tokens.keys():
            self.tokens[session] = self.create_auction_token(session=session)

        # -> Construct token msg
        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_token"
        msg.data = json.dumps(self.tokens[session])
        msg.trace = [self.robot_id] 

        self.team_comms_publisher.publish(msg)

    def create_auction_token(self, session):
        if not DEBUG:
            task_item, task_id, task_value, task_priority = self.find_task_to_auction(self.current_simulation_time)
            return {
                "session": session,
                "task_item": task_item,
                "task_id": task_id,
                "holder_id": self.robot_id,
                "priority": task_priority,
                "value": task_value
            }

        else:
            return {
                "session": session,
                "task_item": "Task 1",
                "task_id": "AAAAAAAA",
                "holder_id": self.robot_id,
                "priority": 0,
                "value": random.randint(0, 10)
            }

    # -------------------------------------------------- Misc
    def handle_msg_trace(self, msg):
        """
        The trace ensures every msg is only broadcasted once per robot
        The second output (trace_flag), returns wether or not a msg needs to be broadcasted
        based on the trace
        """
        if self.robot_id not in msg.trace:
            msg.trace.append(self.robot_id)
            return msg, True
        
        else:
            return msg, False

def main(args=None):
    # `rclpy` library is initialized
    rclpy.init(args=args)

    path_sequence = RLB_mba_agent()

    rclpy.spin(path_sequence)

    path_sequence.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
