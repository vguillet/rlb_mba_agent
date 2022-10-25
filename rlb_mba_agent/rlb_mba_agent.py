

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
import pprint

import sys
from unittest import result
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import JointState, LaserScan
from rlb_utils.msg import Goal, TeamComm, CommsState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from caf_messages.msg import (
    Award, BidStamped, Bid, BundleDescription,
    ItemDescription, AnnouncementStamped, AuctionHeader,
    Header, Results, ResultsStamped)
from caf_essential.utils import utils

import numpy as np
import pandas as pd
import math

from patrol_robot.robot_member_function import Robot

from rlb_tools.Caylus_map_loader import load_maps
from rlb_tools.Raster_ray_tracing import check_comms_available
from rlb_config.simulation_parameters import *

##################################################################################################################


DEBUG = False

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

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
            )

        self.sim_clock_subscriber = self.create_subscription(
            msg_type=TeamComm,
            topic="/RLB_clock",
            callback=self.sim_clock_callback,
            qos_profile=qos 
        )

        # ----------------------------------- Goals publisher
        qos = QoSProfile(depth=10)
        
        # Goals publisher
        self.goal_sequence_publisher = self.create_publisher(
            msg_type=Goal,
            topic='/goals_backlog',
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
        self.sim_time = float(msg.data)

    def process_msg(self, msg):       
        """
        Three checks are performed:
            - If the msg belongs to one of the handled msg types
            - If the msg type corresponds to the current auction phase
            - If the msg has already been broadcasted once by the agent (using trace)

        Msgs that fail any is not rebroadcased
        """ 

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
            self.get_logger().info(f"{self.robot_id}: Running auction, phase = Consensus")
            self.share_current_auction_token()

        elif self.current_auction_phase == "biding":
            self.get_logger().info(f"{self.robot_id}: Running auction, phase = Biding")
            if self.current_session_id in self.tokens.keys():
                self.get_logger().info(f"{self.robot_id}: Session={str(self.current_session_id)[-5:]} - Token value={self.tokens[self.current_session_id]['value']} (task id: {self.tokens[self.current_session_id]['task_id']})")
                # self.get_logger().info(f"\n{pprint.pformat(self.tokens)}\n")
                self.share_current_auction_bid()
        
        else:
            self.get_logger().info(f"{self.robot_id}: Running auction, phase = Award")
            if self.current_session_id in self.tokens.keys():
                if self.tokens[self.current_session_id]['holder_id'] == self.robot_id:
                    self.share_current_auction_award()

    # -------------------------------------------------- Auction award
    def check_auction_award(self, msg):
        award = json.loads(msg.data)
        session = self.current_session_id

        # -> If bid is of current auction
        if award["session"] == session:
            if award['holder_id'] == self.robot_id:
                award = json.loads(msg.data)
                
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

                # -> Assign reward to self
                self.auction_scheme.result_cb(results_stamped_msg)

                # -> Emit goto goal
                # -> Create message
                target_position = json.loads(self.tokens[session]['item_data'])["target_position"]
                point = Point()
                point.x = target_position["x"]
                point.y = target_position["y"]
                point.z = target_position["z"]

                goal = Goal(
                    robot_id=self.robot_id,
                    goal_sequence_id=self.tokens[session]["task_id"],
                    meta_action="",
                    priority=0.,
                    sequence=[point]
                )
                
                if DEBUG: self.get_logger().info(f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                if DEBUG: self.get_logger().info(f"++ Goal sequence {goal.goal_sequence_id} emitted: {goal.sequence} for {goal.robot_id} ++")
                if DEBUG: self.get_logger().info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # -> Publish instruction msg to robot
                self.goal_sequence_publisher.publish(msg=goal)

    def share_current_auction_award(self):
        # -> Construct award msg
        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_award"

        auction_award = self.create_auction_award()

        if auction_award is not None:
            msg.data = json.dumps(auction_award)
            msg.trace = [self.robot_id] 

            self.team_comms_publisher.publish(msg)

    def create_auction_award(self):
        session = self.current_session_id
        
        # Create bid msg list
        if not DEBUG:
            if session not in self.auction_bids.keys():
                self.get_logger().info(f"!!!!!!!!!!! No bids registered for current session {session}")
                return None

            # -> Reconstruct msg from token
            task_item = ItemDescription()
            task_item.item_id = str(self.tokens[session]["item_id"])
            task_item.item_name = self.tokens[session]["item_name"]
            task_item.item_type = self.tokens[session]["item_type"]
            task_item.item_data = self.tokens[session]["item_data"]
            
            # Create announcement stamped msg
            msg = AnnouncementStamped()
            # fill stamp
            msg.header = utils.fill_header(self.robot_id)
            # fill auction header
            msg.body_msg.auction_header.auction_id = \
                'auction_'+self.robot_id+'_'+str(
                    self.auction_cpt)
            # Define deadline
            # TODO: remove this if never used
            # self.auction_duration seconds (int)
            auction_duration = 0.0  
            msg.body_msg.auction_deadline = \
                self.get_clock().now().to_msg()
            msg.body_msg.auction_deadline.nanosec += int(
                auction_duration*1e9)
            msg.body_msg.item_list = [task_item]
            self.get_logger().info(
                f"Announcement msg with sensors {msg}")
            # Store this auction for robot as "auctioneer" role
            self.auction_scheme.current_auction_auctioneer = {
                'announcement_st_msg': msg, 'bids_st_msgs': [],
                'bid_opening': True}
            self.auction_scheme.auction_awaiting_acknowledgements = \
                self.auction_scheme.current_auction_auctioneer
                
            bids = self.auction_bids[session]
            best_bidder, best_bid_val, _ = self.auction_scheme.solve_wdp(bids)

            # -> Construct Award msg
            award_msg = Award()
            award_msg.bidder_id = best_bidder
            
            # -> Construct BundleDescription msg
            bundle_description_msg = BundleDescription()
            bundle_description_msg.bundle_id = self.tokens[session]["task_id"]

            bundle_description_msg.item_bundle = [task_item]

            # -> Construct header msg
            header_msg = Header()
            header_msg.sender_id = self.robot_id
            
            # -> Construct AuctionHeader msg
            auction_header_msg = AuctionHeader()
            auction_header_msg.auction_id = str(session)

            # -> Construct Results msg
            results_msg = Results()
            results_msg.auction_header = auction_header_msg
            results_msg.award_list = [award_msg]
            results_msg.bundle_descriptions = [bundle_description_msg]
            
            # -> Construct ResultsStamped msg
            results_stamped_msg = ResultsStamped()
            results_stamped_msg.header = header_msg
            results_stamped_msg.body_msg = results_msg
            
            # -> Assign reward to self as auctioneer
            if best_bidder == self.robot_id:
                self.auction_scheme.auctioneer_self_award(
                    results_stamped_msg,
                    best_bid_val,
                    self.tokens[session]["task_id"])

                # -> Emit goto goal
                # -> Create message
                target_position = json.loads(self.tokens[session]['item_data'])["target_position"]
                point = Point()
                point.x = target_position["x"]
                point.y = target_position["y"]
                point.z = target_position["z"]

                goal = Goal(
                    robot_id=self.robot_id,
                    goal_sequence_id=self.tokens[session]["task_id"],
                    meta_action="",
                    priority=0.,
                    sequence=[point]
                )
                
                if DEBUG: self.get_logger().info(f"\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                if DEBUG: self.get_logger().info(f"++ Goal sequence {goal.goal_sequence_id} emitted: {goal.sequence} for {goal.robot_id} ++")
                if DEBUG: self.get_logger().info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                # -> Publish instruction msg to robot
                self.goal_sequence_publisher.publish(msg=goal)

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

                    # -> Pop current auctioned task
                    self.auctioneer_auction_to_start.pop(0)
                    if DEBUG: self.get_logger().info(f"{self.robot_id}: Length of auctions to start: {len(self.auctioneer_auction_to_start)}")

                # -> Check if current auction entry exists in self.auction_bids
                if bid not in self.auction_bids[session]:
                    # -> Construct header msg
                    header_msg = Header()
                    header_msg.sender_id = self.robot_id

                    # -> Construct AuctionHeader msg
                    auction_header_msg = AuctionHeader()
                    auction_header_msg.auction_id = str(session)

                    # -> Construct BundleDescription msg
                    bundle_description_msg = BundleDescription()
                    bundle_description_msg.bundle_id = bid["task_id"]

                    # -> Create bid msg
                    bid_msg = Bid()
                    bid_msg.auction_header = auction_header_msg
                    bid_msg.bundle_description = bundle_description_msg
                    bid_msg.bid = [bid["value"]]

                    # -> Created bid stamped msg
                    bid_stamped_msg = BidStamped()
                    bid_stamped_msg.header = header_msg
                    bid_stamped_msg.body_msg = bid_msg

                    self.auction_bids[session].append(bid_stamped_msg)
                    if DEBUG: self.get_logger().info(f"{self.robot_id}: Registered bid for {session}")

                else:
                    # Bid already registered, pass
                    if DEBUG: self.get_logger().info(f"{self.robot_id}: Bid already registered, pass")
                    pass

            else:
                # Not auction token holder, pass
                if DEBUG: self.get_logger().info(f"{self.robot_id}: Not auction token holder, pass")
                pass
    
    def share_current_auction_bid(self):
        # -> Construct bid msg
        msg = TeamComm()

        msg.source = self.robot_id
        msg.source_type = "robot"
        msg.target = "all"
        msg.type = "auction_bid"

        bid = self.create_auction_bid()

        # self.get_logger().info(f"{self}: ------------------- Check sending out auction bid: {bid}")
        if bid is not None:
            # self.get_logger().info(f"{self}: ------------------- Sending out auction bid")
            msg.data = json.dumps(bid)
            msg.trace = [self.robot_id] 

            self.team_comms_publisher.publish(msg)

    def create_auction_bid(self):
        session = self.current_session_id

        task_item = self.tokens[session]["task_item"]

        if not DEBUG:
            if task_item is False:
                return None

            # -> Reconstruct msg from token
            task_item = ItemDescription()
            task_item.item_id = self.tokens[session]["item_id"]
            task_item.item_name = self.tokens[session]["item_name"]
            task_item.item_type = self.tokens[session]["item_type"]
            task_item.item_data = self.tokens[session]["item_data"]

            task_value = self.auction_scheme.compute_bid(task_item)[0]

            # print(f"{self.robot_id}: task value={task_value}")

            return {
                "session": self.current_session_id,
                "task_id": self.tokens[session]["task_id"],
                "holder_id": self.robot_id,
                "priority": 0,
                "value": task_value,
                                
                "task_item": True,
                "item_id": task_item.item_id,
                "item_name": task_item.item_name,
                "item_type": task_item.item_type,
                "item_data": task_item.item_data
            }

        else:
            return {
                "session": self.current_session_id,
                "task_item": task_item,
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

            self.get_logger().info(f"==========================================> {self.robot_id}:  Task item to auction: {len(self.auctioneer_auction_to_start)}")

            return {
                "session": session,
                "task_id": task_id,
                "holder_id": self.robot_id,
                "priority": task_priority,
                "value": task_value,
                
                "task_item": task_item is not None,
                "item_id": task_item.item_id if task_item is not None else None,
                "item_name": task_item.item_name if task_item is not None else None,
                "item_type": task_item.item_type if task_item is not None else None,
                "item_data": task_item.item_data if task_item is not None else None
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
