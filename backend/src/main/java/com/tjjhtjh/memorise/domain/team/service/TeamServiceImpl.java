package com.tjjhtjh.memorise.domain.team.service;

import com.tjjhtjh.memorise.domain.team.repository.TeamRepository;
import com.tjjhtjh.memorise.domain.team.repository.TeamUserRepository;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import com.tjjhtjh.memorise.domain.team.repository.entity.TeamUser;
import com.tjjhtjh.memorise.domain.team.service.dto.request.CreateTeamRequest;
import com.tjjhtjh.memorise.domain.user.exception.NoUserException;
import com.tjjhtjh.memorise.domain.user.repository.UserRepository;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Slf4j
@RequiredArgsConstructor
@Transactional(readOnly = true)
@Service
public class TeamServiceImpl implements TeamService {

    private final TeamRepository teamRepository;
    private final UserRepository userRepository;
    private final TeamUserRepository teamUserRepository;

    private static final String NO_USER = "해당 유저가 존재하지 않습니다";

    @Transactional
    public void createTeam(CreateTeamRequest createTeamRequest) {
        User owner = userRepository.findByUserSeqAndIsDeletedFalse(createTeamRequest.getOwner()).orElseThrow(() -> new NoUserException(NO_USER));
        Team team = (createTeamRequest.getPassword() == null) ? new Team(createTeamRequest.getName(), owner.getUserSeq()) : new Team(createTeamRequest.getName(), owner.getUserSeq(), createTeamRequest.getPassword());
        teamRepository.save(team);
        log.info("teamSeq : {}", team.getTeamSeq());

        teamUserRepository.save(new TeamUser(team, owner));
        for (Long userSeq : createTeamRequest.getUsers()) {
            teamUserRepository.save(new TeamUser(team, userRepository.findByUserSeqAndIsDeletedFalse(userSeq).orElseThrow(() -> new NoUserException(NO_USER))));
        }
    }

}
