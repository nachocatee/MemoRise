package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.team.exception.NotMemberOfGroup;
import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.repository.TeamUserRepository;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;
import com.tjjhtjh.memorise.domain.team.service.dto.request.CreateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.response.TeamDetailResponse;
import com.tjjhtjh.memorise.domain.team.exception.NoTeamException;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TeamServiceImpl implements TeamService {

    private final TeamRepository teamRepository;
    private final UserRepository userRepository;
    private final TeamUserRepository teamUserRepository;

    private static final String NO_USER = "회원 정보가 존재하지 않습니다";
    private static final String NO_TEAM = "팀 정보가 존재하지 않습니다";
    private static final String NOT_MEMBER = "그룹의 멤버가 아닙니다";

    @Transactional
    public void createTeam(CreateTeamRequest createTeamRequest) {
        User owner = userRepository.findByUserSeqAndIsDeletedFalse(createTeamRequest.getOwner()).orElseThrow(() -> new NoUserException(NO_USER));
        Team team = (createTeamRequest.getPassword() == null) ? new Team(createTeamRequest.getName(), owner.getUserSeq()) : new Team(createTeamRequest.getName(), owner.getUserSeq(), createTeamRequest.getPassword());
        teamRepository.save(team);
        teamUserRepository.save(new TeamUser(team, owner));
    }

    @Override
    public TeamDetailResponse getTeamDetailInfo(Long teamSeq, Long userSeq) {
        Team team = teamRepository.findById(teamSeq).orElseThrow(() -> new NoTeamException(NO_TEAM));

        UserInfoResponse me = new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER)));
        UserInfoResponse owner = new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(team.getOwner()).orElseThrow(() -> new NoUserException(NO_USER)));

        List<Long> userSeqs = teamUserRepository.findTeamUserSeqByTeamSeq(teamSeq);
        if (!userSeqs.contains(userSeq)) {
            throw new NotMemberOfGroup(NOT_MEMBER);
        }
        List<UserInfoResponse> members = new ArrayList<>();
        for(Long user : userSeqs) {
            if (!user.equals(userSeq) && !user.equals(team.getOwner())) {
                members.add(new UserInfoResponse(userRepository.findByUserSeqAndIsDeletedFalse(user).orElseThrow(() -> new NoUserException(NO_USER))));
            }
        }

        return new TeamDetailResponse(team, me, owner, members);
    }

}