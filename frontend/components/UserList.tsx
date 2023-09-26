import React from "react";
import { View, Text, StyleSheet, Image } from "react-native";

import Colors from "../constants/colors";
import { calculateDynamicWidth } from "../constants/dynamicSize";

type UserListProps = {
  profileUri: string;
  nickname: string;
  email: string;
};

const UserList: React.FC<UserListProps> = ({ profileUri, nickname, email }) => {
  return (
    <View style={styles.memberInnerContainer}>
      <View style={styles.memberImageContainer}>
        <Image
          style={styles.memberImagebg}
          source={require("../assets/image/profile_bg_square.png")}
        />
        <Image style={styles.memberImage} source={{ uri: profileUri }} />
      </View>
      <View>
        <Text style={styles.memberText}>{nickname}</Text>
        <Text style={styles.memberEmail}>{email}</Text>
      </View>
    </View>
  );
};

export default UserList;

const styles = StyleSheet.create({
  memberInnerContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  memberImageContainer: {
    elevation: 4,
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(15),
  },
  memberImagebg: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
  },
  memberImage: {
    width: calculateDynamicWidth(60),
    height: calculateDynamicWidth(60),
    borderRadius: calculateDynamicWidth(15),
    position: "absolute",
  },
  memberText: {
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(20),
    color: Colors.text,
    marginLeft: calculateDynamicWidth(18),
  },
  memberEmail: {
    fontFamily: "Pretendard-Light",
    fontSize: calculateDynamicWidth(13),
    color: Colors.text,
    marginLeft: calculateDynamicWidth(18),
    opacity: 0.5,
  },
});
