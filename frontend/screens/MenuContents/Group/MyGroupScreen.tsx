import { useEffect, useState, useRef } from "react";
import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import axios from "axios";
import {
  View,
  Text,
  Image,
  Pressable,
  ScrollView,
  Animated,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";
import SmallBtn from "../../../components/Button/SmallBtn";
import GroupBox from "../../../components/GroupBox";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import { styles } from "./GroupStyle";

type RootStackParamList = {
  FindGroup: undefined;
  GroupDetail: GroupDetailParams;
};

type GroupData = [
  {
    teamSeq: number;
    teamName: string;
    myProfile: string;
    ownerProfile: string;
    memberProfiles: [string];
    owner: boolean;
  }
];

type GroupDetailParams = {
  teamSeq: number;
  userSeq: number;
};

const MyGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const [groupData, setGroupData] = useState<GroupData | null>(null);
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 그룹 상세 페이지로 이동하기
  const goDetailHandler = (teamSeq: number) => {
    if (isEditGroup) {
      console.log("나중에 그룹 나가기 함수 넣을거임");
    } else {
      navigation.navigate("GroupDetail", {
        teamSeq: teamSeq,
        userSeq: userId, // 더미 데이터
      });
    }
  };

  // 내 그룹 목록 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/user/${userId}/my-teams`,
          // url: BACKEND_URL + `/user/23/my-teams`, // 더미 데이터
          // url: BACKEND_URL + `/user/26/my-teams`, // 더미 데이터
        });
        setGroupData(res.data);
        console.log("조회 성공");
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

  //  그룹 편집 버튼
  const [isEditGroup, setEditGroup] = useState(false);

  const EditGroupsHandler = () => {
    setEditGroup(!isEditGroup);
  };

  // 그룹 편집 애니메이션
  const shakeAnimation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (isEditGroup) {
      startShake();
    } else {
      shakeAnimation.stopAnimation();
      shakeAnimation.setValue(0);
    }
  }, [isEditGroup]);

  const startShake = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(shakeAnimation, {
          toValue: 1,
          duration: 100,
          useNativeDriver: true,
        }),
        Animated.timing(shakeAnimation, {
          toValue: -1,
          duration: 100,
          useNativeDriver: true,
        }),
      ]),
      {
        iterations: -1, // 무한 반복
      }
    ).start();
  };

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>내 그룹</Text>
          {groupData && groupData[0] && (
            <>
              {isEditGroup ? (
                <SmallBtn onPress={EditGroupsHandler}>완료</SmallBtn>
              ) : (
                <SmallBtn onPress={EditGroupsHandler}>편집</SmallBtn>
              )}
            </>
          )}
        </View>
        {groupData && groupData[0] ? (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
          >
            {groupData.map((group, idx) => (
              <Animated.View
                key={idx}
                style={{
                  transform: [
                    {
                      rotate: shakeAnimation.interpolate({
                        inputRange: [-1, 1],
                        outputRange: ["-1deg", "1deg"], // 여기서 각도 조절
                      }),
                    },
                  ],
                }}
              >
                <GroupBox
                  teamName={group.teamName}
                  myProfile={group.myProfile}
                  memberProfiles={group.memberProfiles}
                  owner={group.owner}
                  goDetailHandler={goDetailHandler}
                  teamSeq={group.teamSeq}
                />
                {isEditGroup && (
                  <Pressable style={styles.deleteContainer}>
                    <Image
                      style={styles.delete}
                      source={require("../../../assets/image/delete.png")}
                    />
                  </Pressable>
                )}
              </Animated.View>
            ))}
          </ScrollView>
        ) : (
          <View style={styles.emptyContainer}>
            <Image
              style={styles.group}
              source={require("../../../assets/icons/group.png")}
            />
            <Text style={styles.empty}>참여한 그룹이 없습니다</Text>
            <Pressable
              style={({ pressed }) =>
                pressed
                  ? [styles.findGroupBtn, styles.pressed]
                  : styles.findGroupBtn
              }
              onPress={() => navigation.navigate("FindGroup")}
            >
              <Text style={styles.btnText}>그룹 찾기</Text>
            </Pressable>
          </View>
        )}
      </View>
    </LinearGradient>
  );
};

export default MyGroupScreen;
